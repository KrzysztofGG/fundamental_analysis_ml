
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from edgar import *

import time
import random
import os
import json
import requests

load_dotenv(override=True)

IDENTITY = os.getenv("IDENTITY")

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {
    "User-Agent": f"financial-research-masters-degree ({IDENTITY})",
    "Accept": "application/json"
}

PRICE_CACHE_DIR = Path("price_cache")
PRICE_CACHE_DIR.mkdir(exist_ok=True)

EDGAR_DATA = "edgar_data"

FORM_FALLBACK_DAYS = {
    '10-Q': 45,
    '10-K': 75,
}

FIRST_USABLE_DATE = '2009-01-01'


def get_company_tickers():
    if os.path.exists('company_tickers.json'):
        with open('company_tickers.json', 'r') as f:
            data = json.load(f)
    else:
        r = requests.get(COMPANY_TICKERS_URL, headers=SEC_HEADERS)
        r.raise_for_status()
        data = r.json()
    return data

def apply_accounting_logic(metrics):
    # ── 0. Shares ──────────────────────────────────────────
    # if metrics.get('commonStockSharesOutstanding') is None:
    #     shares_basic_and_dil = metrics.get('commonStockSharesOutstandingBasicAndDiluted')
    #     shares_dil = metrics.get('commonStockSharesOutstandingDiluted')
    #     if shares_basic_and_dil is not None and shares_dil is not None:
    #         metrics['commonStockSharesOutstanding'] = shares_basic_and_dil - shares_dil

    if metrics.get('commonStockSharesOutstandingDiluted') is None:
        shares_basic_and_dil = metrics.get('commonStockSharesOutstandingBasicAndDiluted')
        shares_basic = metrics.get('commonStockSharesOutstanding')
        if shares_basic_and_dil is not None and shares_basic is not None:
            metrics['commonStockSharesOutstandingDiluted'] = shares_basic_and_dil - shares_basic
            
    # ── 1. Balance sheet identities ──────────────────────────────────────────

    # Total Liabilities from the accounting equation
    if metrics.get('totalLiabilities') is None:
        total_sum = metrics.get('liabilitiesAndStockholdersEquity')
        equity    = metrics.get('totalShareholderEquity')
        if total_sum is not None and equity is not None:
            metrics['totalLiabilities'] = total_sum - equity

    # Shareholder equity from accounting equation (reverse direction)
    if metrics.get('totalShareholderEquity') is None:
        total_sum = metrics.get('liabilitiesAndStockholdersEquity')
        liab      = metrics.get('totalLiabilities')
        if total_sum is not None and liab is not None:
            metrics['totalShareholderEquity'] = total_sum - liab

    # Net working capital

    if metrics.get('totalCurrentAssets') is None:
        assets = metrics.get('totalAssets')
        nonc_assets = metrics.get('totalNonCurrentAssets')

        if assets is not None and nonc_assets is not None:
            metrics['totalCurrentAssets'] = assets - nonc_assets

    if metrics.get('totalCurrentLiabilities') is None:
        liab = metrics.get('totalLiabilities')
        nonc_liab = metrics.get('totalNonCurrentLiabilities')

        if liab is not None and nonc_liab is not None:
            metrics['totalCurrentLiabilities'] = liab - nonc_liab

    ca = metrics.get('totalCurrentAssets')
    cl = metrics.get('totalCurrentLiabilities')
    if ca is not None and cl is not None :
        metrics['netWorkingCapital'] = ca - cl

    # ── 2. Income statement derivations ─────────────────────────────────────

    # ── operatingExpenses = SGA + RD + COGS (last resort) ─────────────────
    if metrics.get('operatingExpenses') is None:
        sga  = metrics.get('sellingGeneralAndAdministrative')
        rd   = metrics.get('researchAndDevelopment')
        cogs = metrics.get('costofGoodsAndServicesSold')
        parts = [v for v in [sga, rd, cogs] if v is not None]
        if parts:
            metrics['operatingExpenses'] = sum(parts)

    # Operating Income fallback
    if metrics.get('operatingIncome') is None:
        gp = metrics.get('grossProfit')
        oe = metrics.get('operatingExpenses')
        if gp is not None and oe is not None:
            metrics['operatingIncome'] = gp - oe

    # ── incomeTaxExpense = current + deferred ──────────────────────────────
    if metrics.get('incomeTaxExpense') is None:
        current = metrics.get('currentIncomeTaxExpense')   # you'd need to scrape these
        deferred = metrics.get('deferredIncomeTaxExpense')  # as separate keys in METRICS
        if current is not None and deferred is not None:
            metrics['incomeTaxExpense'] = current + deferred
        elif current is not None:
            metrics['incomeTaxExpense'] = current
        elif deferred is not None:
            metrics['incomeTaxExpense'] = deferred

    if metrics.get('operatingIncome') is None:
        non_op = metrics.get('nonoperatingIncome')
        ni = metrics.get('netIncome')
        ite = metrics.get('incomeTaxExpense')
        ie = metrics.get('interestExpense')
        if all(v is not None for v in [non_op, ni, ite, ie]):
            metrics['operatingIncome'] = ni + ite + ie - non_op

    # D&A as sum of Depreciation and Ammortization
    if metrics.get('depreciationAndAmortization') is None:
        d = metrics.get('depreciation')
        am = metrics.get('amortization')

        if d is not None and am is not None:
            metrics['depreciationAndAmortization'] = d + am
        elif d is not None:
            metrics['depreciationAndAmortization'] = d
        elif am is not None:
            metrics['depreciationAndAmortization'] = am

    # EBIT: Operating Income is effectively EBIT for most companies
    if metrics.get('ebit') is None:
        metrics['ebit'] = metrics.get('operatingIncome')

    # EBITDA = EBIT + D&A
    if metrics.get('ebitda') is None:
        ebit = metrics.get('ebit') or metrics.get('operatingIncome')
        da   = metrics.get('depreciationAndAmortization')
        if ebit is not None and da is not None:
            metrics['ebitda'] = ebit + da  

    # EBITDA fallback: Net Income + Tax + Interest + D&A
    if metrics.get('ebitda') is None:
        ni       = metrics.get('netIncome')
        tax      = metrics.get('incomeTaxExpense')
        interest = metrics.get('interestExpense')
        da       = metrics.get('depreciationAndAmortization')
        if all(v is not None for v in [ni, tax, interest, da]):
            metrics['ebitda'] = ni + tax + interest + da

    # COGS from Revenue - Gross Profit
    if metrics.get('costofGoodsAndServicesSold') is None:
        rev = metrics.get('totalRevenue')
        gp = metrics.get('grossProfit') 
        if rev is not None and gp is not None:
            metrics['costofGoodsAndServicesSold'] = rev - gp

    # Gross Profit from Revenue - COGS (reverse direction)
    if metrics.get('grossProfit') is None:
        rev  = metrics.get('totalRevenue')
        cogs = metrics.get('costofGoodsAndServicesSold')
        if rev is not None and cogs is not None:
            metrics['grossProfit'] = rev - cogs
    
    if metrics.get('grossProfit') is None:
        oe = metrics.get('operatingExpenses')
        oi = metrics.get('operatingIncome')
        if oe is not None and oi is not None:
            metrics['grossProfit'] = oe + oi
        

    # EBT (Earnings Before Tax) = Net Income + Tax Expense
    if metrics.get('earningsBeforeTax') is None:
        ni  = metrics.get('netIncome')
        tax = metrics.get('incomeTaxExpense')
        if ni is not None and tax is not None:
            metrics['earningsBeforeTax'] = ni + tax
    
    # ── 3. Cash flow derivations ──────────────────────────────────────────────

    # Free Cash Flow = Operating CF - CapEx
    if metrics.get('freeCashFlow') is None:
        opf   = metrics.get('operatingCashflow')
        capex = metrics.get('capitalExpenditures')
        if opf is not None and capex is not None:
            # CapEx is usually stored as negative in filings; normalise
            metrics['freeCashFlow'] = opf - abs(capex)

    # ── 4. Debt & leverage aggregates ────────────────────────────────────────

    # Total debt
    if metrics.get('shortLongTermDebtTotal') is None:
        st = metrics.get('shortTermDebt')
        lt = metrics.get('longTermDebt')
        if st is not None and lt is not None:
            metrics['shortLongTermDebtTotal'] = st + lt
        # elif lt is not None:
        #     metrics['shortLongTermDebtTotal'] = lt # partial

    # Net debt = Total Debt - Cash
    if metrics.get('netDebt') is None:
        debt = metrics.get('shortLongTermDebtTotal')
        cash = metrics.get('cashAndCashEquivalentsAtCarryingValue')
        if debt is not None and cash is not None:
            metrics['netDebt'] = debt - cash
    
    # ── 6. Market-based ratios (only if stock price available) ───────────────

    # Market Cap = Shares Count * Share Price
    if metrics.get('marketCap') is None:
        price  = metrics.get('stock_price')
        shares = metrics.get('commonStockSharesOutstanding')
        if price is not None and shares is not None:
            metrics['marketCap'] = price * shares

    # EV = Market Cap + Total Debt - Cash
    if metrics.get('enterpriseValue') is None:
        mc   = metrics.get('marketCap')
        debt = metrics.get('shortLongTermDebtTotal')
        cash = metrics.get('cashAndCashEquivalentsAtCarryingValue')
        if mc is not None and debt is not None and cash is not None:
            metrics['enterpriseValue'] = mc + debt - cash
        # elif mc is not None and debt is not None:
        #     metrics['enterpriseValue'] = mc + debt  # partial, no cash offset

    if metrics.get('dividendPayout') is None:
        dps = metrics.get('commonStockDividendsPerShareCashPaid')
        shares = metrics.get('commonStockSharesOutstanding')
        if dps is not None and shares is not None:
            metrics['dividendPayout'] = abs(dps * shares)
        else:
            metrics['dividendPayout'] = 0.0 # assume no dividend


# def apply_accounting_logic_av(metrics: dict):
#     '''
#     Used to count metrics not available immediately 
#     via alpha vantage request
#     '''
#     # Free Cash Flow = Operating CF - CapEx
#     if metrics.get('freeCashFlow') is None:
#         opf   = metrics.get('operatingCashflow')
#         capex = metrics.get('capitalExpenditures')
#         if opf is not None and capex is not None:
#             # CapEx is usually stored as negative in filings; normalise
#             metrics['freeCashFlow'] = opf - abs(capex)

#         # EBIT: Operating Income is effectively EBIT for most companies
#     if metrics.get('ebit') is None:
#         metrics['ebit'] = metrics.get('operatingIncome')

#     # EBITDA = EBIT + D&A
#     if metrics.get('ebitda') is None:
#         ebit = metrics.get('ebit') or metrics.get('operatingIncome')
#         da   = metrics.get('depreciationAndAmortization')
#         if ebit is not None and da is not None:
#             metrics['ebitda'] = ebit + da  

#     # EBITDA fallback: Net Income + Tax + Interest + D&A
#     if metrics.get('ebitda') is None:
#         ni       = metrics.get('netIncome')
#         tax      = metrics.get('incomeTaxExpense')
#         interest = metrics.get('interestExpense')
#         da       = metrics.get('depreciationAndAmortization')
#         if all(v is not None for v in [ni, tax, interest, da]):
#             metrics['ebitda'] = ni + tax + interest + da

#     # Net debt = Total Debt - Cash
#     if metrics.get('netDebt') is None:
#         debt = metrics.get('shortLongTermDebtTotal')
#         cash = metrics.get('cashAndCashEquivalentsAtCarryingValue')
#         if debt is not None and cash is not None:
#             metrics['netDebt'] = debt - cash

#     # Market Cap = Shares Count * Share Price
#     if metrics.get('marketCap') is None:
#         price  = metrics.get('stock_price')
#         shares = metrics.get('commonStockSharesOutstanding')
#         if price is not None and shares is not None:
#             metrics['marketCap'] = price * shares

#     # EV = Market Cap + Total Debt - Cash
#     if metrics.get('enterpriseValue') is None:
#         mc   = metrics.get('marketCap')
#         debt = metrics.get('shortLongTermDebtTotal')
#         cash = metrics.get('cashAndCashEquivalentsAtCarryingValue')
#         if mc is not None and debt is not None and cash is not None:
#             metrics['enterpriseValue'] = mc + debt - cash

#     # NWC = Current Assets - Current Liabilities
#     ca = metrics.get('totalCurrentAssets')
#     cl = metrics.get('totalCurrentLiabilities')
#     if ca is not None and cl is not None :
#         metrics['netWorkingCapital'] = ca - cl

#     # EBT (Earnings Before Tax) = Net Income + Tax Expense
#     if metrics.get('earningsBeforeTax') is None:
#         ni  = metrics.get('netIncome')
#         tax = metrics.get('incomeTaxExpense')
#         if ni is not None and tax is not None:
#             metrics['earningsBeforeTax'] = ni + tax

def apply_accounting_logic_av(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['freeCashFlow', 'ebit', 'ebitda', 'netDebt', 
            'marketCap', 'enterpriseValue', 'netWorkingCapital', 'earningsBeforeTax']:
        if col not in df.columns:
            df[col] = np.nan

    # Free Cash Flow = Operating CF - CapEx
    df['freeCashFlow'] = df['freeCashFlow'].fillna(
        df['operatingCashflow'] - df['capitalExpenditures'].abs()
    )

    # EBIT ≈ Operating Income
    df['ebit'] = df['ebit'].fillna(df['operatingIncome'])

    # EBITDA = EBIT + D&A
    df['ebitda'] = df['ebitda'].fillna(
        df['ebit'] + df['depreciationAndAmortization']
    )

    # EBITDA fallback = Net Income + Tax + Interest + D&A
    df['ebitda'] = df['ebitda'].fillna(
        df['netIncome'] + df['incomeTaxExpense'] + df['interestExpense'] + df['depreciationAndAmortization']
    )

    # Net Debt = Total Debt - Cash
    df['netDebt'] = df['netDebt'].fillna(
        df['shortLongTermDebtTotal'] - df['cashAndCashEquivalentsAtCarryingValue']
    )

    # Market Cap = Price * Shares
    df['marketCap'] = df['marketCap'].fillna(
        df['stock_price'] * df['commonStockSharesOutstanding']
    )

    # EV = Market Cap + Debt - Cash
    df['enterpriseValue'] = df['enterpriseValue'].fillna(
        df['marketCap'] + df['shortLongTermDebtTotal'] - df['cashAndCashEquivalentsAtCarryingValue']
    )

    # NWC = Current Assets - Current Liabilities
    df['netWorkingCapital'] = df['totalCurrentAssets'] - df['totalCurrentLiabilities']

    # EBT = Net Income + Tax
    df['earningsBeforeTax'] = df['earningsBeforeTax'].fillna(
        df['netIncome'] + df['incomeTaxExpense']
    )

    return df


def sanitize_numeric(value):
    """Convert any value to float, or None if not possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(',', '').strip())
        except (ValueError, AttributeError):
            return None
    return None

def sanitize_metrics(metrics: dict) -> dict:
    """
    Run once after XBRL extraction and RENAME_TABLE mapping,
    before any accounting logic or indicator computation.
    Non-numeric fields are passed through as-is.
    """
    NON_NUMERIC = {'currency_symbol', 'reportedCurrency', 
                   'fiscalDateEnding', 'filing_date', 'filing_date_used', 'ticker'}
    return {
        k: (v if k in NON_NUMERIC else sanitize_numeric(v))
        for k, v in metrics.items()
    }

# def compute_indicators(metrics: dict, prev_metrics: dict | None = None, period: str = 'quarterly') -> dict: 
#     """
#         Step 2: compute all ratio-based indicators from enriched metrics.
#         Requires apply_accounting_logic() to have been called first.
#         Returns a flat dict of indicator values (None if inputs missing).
#     """

#     ind = {}
  
#     def safe_div(a, b):
#         if a is None or b is None or b == 0:
#             return None
#         else:
#             return round(a / b, 6)
        
#     rev   = metrics.get('totalRevenue')
#     ni    = metrics.get('netIncome')
#     gp    = metrics.get('grossProfit')
#     # cogs  = metrics.get('costofGoodsAndServicesSold')
#     assets= metrics.get('totalAssets')
#     eq    = metrics.get('totalShareholderEquity')
#     liab  = metrics.get('totalLiabilities')
#     ca    = metrics.get('totalCurrentAssets')
#     cl    = metrics.get('totalCurrentLiabilities')
#     ocf   = metrics.get('operatingCashflow')
#     capex = metrics.get('capitalExpenditures')
#     fcf   = metrics.get('freeCashFlow')
#     ebit  = metrics.get('ebit')
#     ebitda= metrics.get('ebitda')
#     da    = metrics.get('depreciationAndAmortization')
#     td    = metrics.get('shortLongTermDebtTotal')
#     nd    = metrics.get('netDebt')
#     ev    = metrics.get('enterpriseValue')
#     mc    = metrics.get('marketCap')
#     price = metrics.get('stock_price')
#     shares= metrics.get('commonStockSharesOutstanding')
#     # shares_d = metrics.get('commonStockSharesOutstandingDiluted')
#     re    = metrics.get('retainedEarnings')
#     div   = metrics.get('dividendPayout')
#     nwc   = metrics.get('netWorkingCapital')
#     interest = metrics.get('interestExpense')
#     tax   = metrics.get('incomeTaxExpense')
#     sbc   = metrics.get('stockBasedCompensation')
#     oi    = metrics.get('operatingIncome')
#     cash  = metrics.get('cashAndCashEquivalentsAtCarryingValue')
#     cfi = metrics.get('cashflowFromInvestment')
#     rd    = metrics.get('researchAndDevelopment')
#     sga   = metrics.get('sellingGeneralAndAdministrative')
#     opex  = metrics.get('operatingExpenses')

#     # ── Profitability margins ────────────────────────────────────────────────
#     ind['grossMargin']      = safe_div(gp, rev)
#     ind['operatingMargin']  = safe_div(oi, rev)
#     ind['netProfitMargin']  = safe_div(ni, rev)
#     ind['ebitdaMargin']     = safe_div(ebitda, rev)
#     ind['fcfMargin']        = safe_div(fcf, rev)

#     # ── Efficiency / returns ─────────────────────────────────────────────────
#     ind['roe']  = safe_div(ni, eq)   # Return on Equity
#     ind['roa']  = safe_div(ni, assets)  # Return on Assets
#     ind['equityMultiplier'] = safe_div(assets, eq)

#     # CapEx intensity — how asset-heavy is the business?
#     ind['capexToRevenue']     = safe_div(capex, rev)
#     ind['capexToOcf']         = safe_div(capex, ocf)        # CapEx consumed from operating CF
#     ind['capexToDepreciation']= safe_div(capex, da)         # <1 = underinvesting, >1 = growing

#     # Maintenance vs growth CapEx proxy
#     # If CapEx ≈ D&A → maintenance mode; excess = growth investment
#     if capex is not None and da is not None:
#         ind['growthCapex'] = abs(capex) - abs(da)           # Rough growth CapEx estimate
#     else:
#         ind['growthCapex'] = None

#     # ROIC = EBIT*(1-tax_rate) / Invested Capital
#     # Invested Capital = Total Equity + Total Debt - Cash
#     invested_capital = None
#     if eq is not None and td is not None and cash is not None:
#         invested_capital = eq + td - cash
#     elif eq is not None and td is not None:
#         invested_capital = eq + td
    
#     ebt = metrics.get('earningsBeforeTax')
#     if tax and ebt and ebt > 0:
#         tax_rate = safe_div(tax, ebt) or 0.21
#     else:
#         tax_rate = 0.21
        
#     if ebit is not None and invested_capital is not None:
#         nopat = ebit * (1 - (tax_rate or 0.21))
#         ind['roic'] = safe_div(nopat, invested_capital)
#     else:
#         ind['roic'] = None
#     ind['investedCapital'] = invested_capital

#     # Tax rate — deviations from ~21% can signal aggressive accounting or structuring
#     ind['effectiveTaxRate']   = safe_div(tax, metrics.get('earningsBeforeTax') or metrics.get('incomeBeforeTax'))

#     # ── Cash flow quality ────────────────────────────────────────────────────
#     if ni is not None and ni > 0:
#         ind['cashKingRatio'] = safe_div(ocf, ni)
#     else:
#         ind['cashKingRatio'] = None    # OCF / Net Income (>1 = quality earnings)
#     ind['ocfToRevenue']   = safe_div(ocf, rev)

#     # SBC-adjusted FCF: removes non-cash compensation that dilutes shareholders
#     # This is what most tech analysts mean by "real" FCF
#     if fcf is not None and sbc is not None:
#         ind['fcfAfterSbcDilutionCost'] = fcf - abs(sbc)
#         ind['fcfAfterSbcDilutionCostMargin'] = safe_div(fcf - abs(sbc), rev)
#     else:
#         ind['fcfAfterSbcDilutionCost'] = None
#         ind['fcfAfterSbcDilutionCostMargin'] = None

#     # SBC as % of revenue — signals how much dilution the business is running
#     ind['sbcToRevenue'] = safe_div(sbc, rev)

#     # Owner Earnings = Net Income + D&A - Maintenance CapEx
#     # Heuristic: maintenance CapEx ≈ D&A (conservative proxy when not reported separately)
#     if ni is not None and da is not None and capex is not None:
#         sbc_cost = abs(sbc) if sbc is not None else 0  # treat as 0 if not reported
#         ind['ownerEarnings'] = ni + da - abs(capex) - sbc_cost
#     else:
#         ind['ownerEarnings'] = None

#     # Sloan Ratio = (Net Income - OCF) / Total Assets (accruals quality, <10% healthy)
#     if ni is not None and ocf is not None and cfi is not None and assets:
#         ind['sloanRatio'] = safe_div(ni - ocf - cfi, assets)
#     else:
#         ind['sloanRatio'] = None

#     # ── Liquidity & leverage ─────────────────────────────────────────────────
#     ind['currentRatio']     = safe_div(ca, cl)
#     if ca is not None and cl is not None:
#         ind['quickRatio']       = safe_div(ca - (metrics.get('inventory') or 0), cl)
#     else:
#         ind['quickRatio'] = None
#     ind['debtToEquity']     = safe_div(td, eq)
#     ind['debtToAssets']     = safe_div(td, assets)
#     ind['interestCoverage'] = safe_div(ebit, abs(interest)) if interest else None  # EBIT / Interest (>3 healthy)
#     ind['netDebtToEbitda']  = safe_div(nd, ebitda)
#     ind['debtToEbitda']       = safe_div(td, ebitda)       # Leverage (banks use <3x as healthy)
#     # FCF conversion from EBITDA — high = earnings are cash-backed
#     ind['fcfToEbitda']        = safe_div(fcf, ebitda)

#     # ── Per share ────────────────────────────────────────────────────────────
#     ind['eps']              = safe_div(ni, shares)
#     # ind['epsDiluted']       = safe_div(ni, shares_d)
#     ind['bookValuePerShare']= safe_div(eq, shares)
#     ind['fcfPerShare']      = safe_div(fcf, shares)
#     ind['ocfPerShare']      = safe_div(ocf, shares)
#     ind['revenuePerShare']  = safe_div(rev, shares)

#     # ── Market multiples (need price) ────────────────────────────────────────
#     if mc:
#         ind['peRatio']      = safe_div(price, ind['eps'])
#         ind['pbRatio']      = safe_div(mc, eq)
#         ind['psRatio']      = safe_div(mc, rev)
#         ind['pfcfRatio']    = safe_div(mc, fcf)
#         ind['pCfoRatio']    = safe_div(mc, ocf)
#         ind['evToEbitda']   = safe_div(ev, ebitda)
#         ind['evToFcf']      = safe_div(ev, fcf)
#         ind['evToRevenue']  = safe_div(ev, rev)
#         ind['evToEbit']     = safe_div(ev, ebit)
#     else:
#         for k in ['peRatio','pbRatio','psRatio','pfcfRatio','pCfoRatio',
#                 'evToEbitda','evToFcf','evToRevenue','evToEbit']:
#             ind[k] = None

#     # ── Dividends ────────────────────────────────────────────────────────────
#     # dividendPayout from XBRL is total cash paid; annualise from quarterly
#     if div is not None and price is not None and shares is not None and shares > 0:
#         dps = abs(div) / shares  # dividend per share this quarter
#         if period == 'quarterly':
#             ind['dividendYield']  = safe_div(dps * 4, price) # annualised
#         else: #yearly
#             ind['dividendYield']  = safe_div(dps, price)  
#     else:
#         ind['dividendYield'] = None
#     ind['dividendPayoutRatio']    = safe_div(abs(div) if div is not None else None, ni)

#     if prev_metrics:
#         prev_assets = prev_metrics.get('totalAssets')
#         ind['assetGrowth'] = safe_div(assets - prev_assets, abs(prev_assets)) if assets and prev_assets else None
        
#         prev_ocf = prev_metrics.get('operatingCashflow')
#         ind['ocfGrowth'] = safe_div(ocf - prev_ocf, abs(prev_ocf)) if ocf and prev_ocf else None

#         prev_fcf = prev_metrics.get('freeCashFlow')
#         ind['fcfGrowth'] = safe_div(fcf - prev_fcf, abs(prev_fcf)) if fcf and prev_fcf else None

#         prev_ni = prev_metrics.get('netIncome')
#         ind['niGrowth'] = safe_div(ni - prev_ni, abs(prev_ni)) if ni and prev_ni else None
        
#         prev_rev = prev_metrics.get('totalRevenue')
#         prev_gp = prev_metrics.get('grossProfit')
#         ind['grossMarginDelta'] = (safe_div(gp, rev) or 0) - (safe_div(prev_gp, prev_rev) or 0) \
#             if gp and rev and prev_gp and prev_rev else None  # Margin expansion signal
        
#         prev_eps = prev_metrics.get('eps') or safe_div(
#             prev_metrics.get('netIncome'),
#             prev_metrics.get('commonStockSharesOutstanding')
#         )
#         cur_eps = ind.get('eps')
#         if period == 'quarterly':
#             ind['revenueGrowthQoQ'] = safe_div(rev - prev_rev, abs(prev_rev)) if rev and prev_rev else None
#             ind['epsGrowthQoQ']     = safe_div(cur_eps - prev_eps, abs(prev_eps)) if cur_eps and prev_eps else None
#         else:
#             ind['revenueGrowthYoY'] = safe_div(rev - prev_rev, abs(prev_rev)) if rev and prev_rev else None
#             ind['epsGrowthYoY']     = safe_div(cur_eps - prev_eps, abs(prev_eps)) if cur_eps and prev_eps else None
        
#     else:
#         if period == 'quarterly':
#             ind['revenueGrowthQoQ'] = None
#             ind['epsGrowthQoQ']     = None
#         else:
#             ind['revenueGrowthYoY'] = None
#             ind['epsGrowthYoY']     = None

#     # ── Altman Z-Score (public non-financial companies) ──────────────────────
#     # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
#     if all(v is not None for v in [nwc, re, ebit, mc, liab, rev, assets]) and assets > 0 and liab > 0:
#         A = nwc  / assets           # Working Capital / Total Assets
#         B = re   / assets           # Retained Earnings / Total Assets
#         C = ebit / assets           # EBIT / Total Assets
#         D = mc   / liab             # Market Cap / Total Liabilities
#         E = rev  / assets           # Revenue / Total Assets
#         ind['altmanZScore'] = round(1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E, 4)
#     else:
#         ind['altmanZScore'] = None

#     # ── Rule of 40 (SaaS/growth metric) ─────────────────────────────────────
#     fcf_margin_pct = (ind.get('fcfMargin') or 0) * 100
#     if period == 'quarterly':
#         rev_growth_pct = (ind.get('revenueGrowthQoQ') or 0) * 4 * 100 # annualised %
#         if ind.get('revenueGrowthQoQ') is not None and ind.get('fcfMargin') is not None:
#             ind['ruleOf40'] = round(rev_growth_pct + fcf_margin_pct, 2)
#         else:
#             ind['ruleOf40'] = None
#     else:
#         rev_growth_pct = (ind.get('revenueGrowthYoY') or 0) * 100
#         if ind.get('revenueGrowthYoY') is not None and ind.get('fcfMargin') is not None:
#             ind['ruleOf40'] = round(rev_growth_pct + fcf_margin_pct, 2)
#         else:
#             ind['ruleOf40']= None
    
#     return ind

def compute_indicators(df: pd.DataFrame, period: str = 'quarterly') -> pd.DataFrame:
    ind = pd.DataFrame(index=df.index)

    # ── Shorthand refs ───────────────────────────────────────────────────────
    rev    = df['totalRevenue']
    ni     = df['netIncome']
    gp     = df['grossProfit']
    assets = df['totalAssets']
    eq     = df['totalShareholderEquity']
    liab   = df['totalLiabilities']
    ca     = df['totalCurrentAssets']
    cl     = df['totalCurrentLiabilities']
    ocf    = df['operatingCashflow']
    capex  = df['capitalExpenditures']
    fcf    = df['freeCashFlow']
    ebit   = df['ebit']
    ebitda = df['ebitda']
    da     = df['depreciationAndAmortization']
    td     = df['shortLongTermDebtTotal']
    nd     = df['netDebt']
    ev     = df['enterpriseValue']
    mc     = df['marketCap']
    price  = df['stock_price']
    shares = df['commonStockSharesOutstanding']
    re     = df['retainedEarnings']
    div    = df['dividendPayout']
    nwc    = df['netWorkingCapital']
    interest = df['interestExpense']
    tax    = df['incomeTaxExpense']
    sbc    = df['stockBasedCompensation']
    oi     = df['operatingIncome']
    cash   = df['cashAndCashEquivalentsAtCarryingValue']
    cfi    = df['cashflowFromInvestment']
    ebt    = df['earningsBeforeTax']

    # ── Profitability margins ────────────────────────────────────────────────
    ind['grossMargin']     = gp / rev
    ind['operatingMargin'] = oi / rev
    ind['netProfitMargin'] = ni / rev
    ind['ebitdaMargin']    = ebitda / rev
    ind['fcfMargin']       = fcf / rev

    # ── Efficiency / returns ─────────────────────────────────────────────────
    ind['roe']             = ni / eq
    ind['roa']             = ni / assets
    ind['equityMultiplier']= assets / eq

    # ── CapEx intensity ──────────────────────────────────────────────────────
    ind['capexToRevenue']      = capex / rev
    ind['capexToOcf']          = capex / ocf
    ind['capexToDepreciation'] = capex / da
    ind['growthCapex']         = capex.abs() - da.abs()

    # ── ROIC ─────────────────────────────────────────────────────────────────
    invested_capital = np.where(
        cash.notna(), eq + td - cash,
        np.where(td.notna(), eq + td, np.nan)
    )
    ind['investedCapital'] = invested_capital

    tax_rate = (tax / ebt).clip(0, 1)
    tax_rate = tax_rate.where((ebt.notna()) & (ebt > 0), 0.21)
    nopat = ebit * (1 - tax_rate)
    ind['roic'] = nopat / pd.Series(invested_capital, index=df.index)

    ind['effectiveTaxRate'] = tax / ebt.fillna(df.get('incomeBeforeTax', np.nan))

    # ── Cash flow quality ────────────────────────────────────────────────────
    ind['ocfToRevenue']  = ocf / rev

    ind['reinvestmentRate'] = (capex.abs() + da.abs()) / assets
    ind['organicFcf'] = ocf - capex.abs()
    ind['organicFcfMargin'] = ind['organicFcf'] / rev

    # ── Earnings quality ─────────────────────────────────────────────────────
    ind['earningsQuality'] = ocf / ni.where(ni.abs() > 0)
    ind['grossProfitToAssets'] = gp / assets
    ind['revenueToAssets'] = rev / assets

    # ind['cashKingRatio'] = (ocf / ni).where(ni > 0)

    # ind['fcfAfterSbcDilutionCost']       = fcf - sbc.abs()
    # ind['fcfAfterSbcDilutionCostMargin'] = ind['fcfAfterSbcDilutionCost'] / rev
    # ind['sbcToRevenue']  = sbc / rev

    inventory = df.get('inventory', pd.Series(0, index=df.index)).fillna(0)
    ind['ownerEarnings'] = ni + da - capex.abs() - sbc.abs().fillna(0)
    ind['sloanRatio']    = (ni - ocf - cfi) / assets

    # ── Liquidity & leverage ─────────────────────────────────────────────────
    ind['currentRatio']     = ca / cl
    ind['quickRatio']       = (ca - inventory) / cl
    ind['debtToEquity']     = td / eq
    ind['debtToAssets']     = td / assets
    ind['interestCoverage'] = ebit / interest.abs()
    ind['netDebtToEbitda']  = nd / ebitda
    ind['debtToEbitda']     = td / ebitda
    ind['fcfToEbitda']      = fcf / ebitda

    # ── Per share ────────────────────────────────────────────────────────────
    ind['eps']              = ni / shares
    ind['bookValuePerShare']= eq / shares
    ind['fcfPerShare']      = fcf / shares
    ind['ocfPerShare']      = ocf / shares
    ind['revenuePerShare']  = rev / shares

    # ── Market multiples ────────────────────────────────────────────────────
    ind['peRatio']    = (price / ind['eps']).where(mc.notna())
    ind['pbRatio']    = (mc / eq).where(mc.notna())
    ind['psRatio']    = (mc / rev).where(mc.notna())
    ind['pfcfRatio']  = (mc / fcf).where(mc.notna())
    ind['pCfoRatio']  = (mc / ocf).where(mc.notna())
    ind['evToEbitda'] = (ev / ebitda).where(mc.notna())
    ind['evToFcf']    = (ev / fcf).where(mc.notna())
    ind['evToRevenue']= (ev / rev).where(mc.notna())
    ind['evToEbit']   = (ev / ebit).where(mc.notna())

    # ── Dividends ────────────────────────────────────────────────────────────
    # dps = div.abs() / shares
    # if period == 'quarterly':
    #     ind['dividendYield'] = (dps * 4 / price).where(div.notna() & shares.notna())
    # else:
    #     ind['dividendYield'] = (dps / price).where(div.notna() & shares.notna())

    # ind['dividendPayoutRatio'] = div.abs() / ni
    # ── Dividend alternatives ─────────────────────────────────────────────────
    div_safe = div.fillna(0).abs()
    ind['cashReturnToRevenue'] = div_safe / rev         
    ind['payoutSustainability'] = div_safe / ocf
    # ind['payoutToFcf'] = div_safe / fcf    

    # ── Growth (lagged) ──────────────────────────────────────────────────────
    growth_suffix = 'QoQ' if period == 'quarterly' else 'YoY'

    ind['assetGrowth']    = (assets - assets.shift(1)) / assets.shift(1).abs()
    ind['ocfGrowth']      = (ocf - ocf.shift(1)) / ocf.shift(1).abs()
    ind['fcfGrowth']      = (fcf - fcf.shift(1)) / fcf.shift(1).abs()
    ind['niGrowth']       = (ni - ni.shift(1)) / ni.shift(1).abs()
    ind['grossMarginDelta'] = ind['grossMargin'] - (gp.shift(1) / rev.shift(1))

    eps = ind['eps']
    ind[f'revenueGrowth{growth_suffix}'] = (rev - rev.shift(1)) / rev.shift(1).abs()
    ind[f'epsGrowth{growth_suffix}']     = (eps - eps.shift(1)) / eps.shift(1).abs()

    # ── Altman Z-Score ───────────────────────────────────────────────────────
    A = nwc / assets
    B = re / assets
    C = ebit / assets
    D = mc / liab
    E = rev / assets
    ind['altmanZScore'] = (1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E).where(
        assets.notna() & liab.notna() & mc.notna() & (assets > 0) & (liab > 0)
    )

    # ── Rule of 40 ───────────────────────────────────────────────────────────
    fcf_margin_pct = ind['fcfMargin'] * 100
    rev_growth_col = f'revenueGrowth{growth_suffix}'
    if period == 'quarterly':
        rev_growth_pct = ind[rev_growth_col] * 4 * 100
    else:
        rev_growth_pct = ind[rev_growth_col] * 100


    ind['ruleOf40'] = (rev_growth_pct + fcf_margin_pct).where(
        ind[rev_growth_col].notna() & ind['fcfMargin'].notna()
    )

    ind = ind.replace([np.inf, -np.inf], np.nan).round(6)

    # return ind.add_prefix('ind_')
    return ind


def save_metadata(ticker: str, metadata: dict, path: str):
    os.makedirs(EDGAR_DATA, exist_ok=True)

    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    
    existing[ticker] = metadata
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)

def get_company_metadata(ticker: str, retries: int = 3) -> dict:

    for attempt in range(retries):
        try:
            info = yf.Ticker(ticker).info
            return {
                "sector": info.get("sector", None),
                "industry": info.get("industry", None),
            }
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"[{ticker}] metadata attempt {attempt+1} failed: {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
        return {"sector": None, "industry": None, "marketCap": None}


def get_price_history(ticker: str, retries: int = 3) -> pd.DataFrame | None:
    cache_path = PRICE_CACHE_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period='max')
            hist.index = hist.index.tz_localize(None)
            hist.to_parquet(cache_path)
            return hist
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"[{ticker}] price history attempt {attempt+1} failed: {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
    return None
    
def lookup_price(price_history: pd.DataFrame, date_str: str, window_days: int = 5) -> float | None:
    if price_history is None or not date_str or str(date_str) == "None":
        return None
    
    try:
        target = pd.Timestamp(str(date_str)[:10])
        end = target + pd.Timedelta(days=window_days)
        subset = price_history[target:end]
        if subset.empty:
            return None
        return round(subset['Close'].iloc[0], 4)
    except Exception as e:
        print(f"Failed to collect subset of price history for date {date_str}: {e}")
        return None
    
def lookup_prices_vectorized(price_history: pd.DataFrame, dates: pd.Series, window_days: int = 5) -> pd.Series:
    if price_history is None:
        return pd.Series(None, index=dates.index)

    closes = price_history['Close'].copy()
    closes = closes[~closes.index.duplicated(keep='last')]

    target_dates = pd.to_datetime(dates.values)

    if pd.DatetimeIndex(target_dates).duplicated().any():
        dupes = target_dates[pd.DatetimeIndex(target_dates).duplicated(keep=False)]
        raise ValueError(f"Duplicate dates in input: {dupes.tolist()}")
    
    looked_up = closes.reindex(closes.index.union(target_dates)) \
                 .ffill(limit=window_days) \
                 .reindex(target_dates) \
                 .round(4)
    
    return pd.Series(looked_up.values, index=dates.index)
                 
    

def get_report_dates(ticker: str, period: str) -> pd.DataFrame | None:
    """
    period: 'quarterly' | 'yearly'
    """

    try:
        company = Company(ticker)
    except CompanyNotFoundError as e:
        print(f"Ticker {ticker} not found in EDGAR: {str(e)}")
        return None
    
    if company.is_foreign:
        forms_req = ['20-F']
    else:
        if period == 'yearly':
            forms_req = ['10-K']
        elif period == 'quarterly':
            forms_req = ['10-K', '10-Q']
    
    df = company.get_filings(form=forms_req) \
                .to_pandas()[['filing_date', 'reportDate', 'form']]
    df = df[df['form'].isin(forms_req)]
    # if period == 'yearly':
        # df = company.get_filings(form='10-K') \
        #     .to_pandas()[['filing_date', 'reportDate', 'form']] 
        # df = df[df['form'] == '10-K']
    # elif period == 'quarterly':
        # df = company.get_filings(form=["10-K", "10-Q"]) \
        #     .to_pandas()[['filing_date', 'reportDate', 'form']]
        # df = df[df['form'].isin(['10-K', '10-Q'])]

    
    df = df \
        .sort_values('filing_date', ascending=True) \
        .drop_duplicates(subset='reportDate', keep='first') \
        .sort_values('reportDate', ascending=True) \
        .reset_index(drop=True)
    
    df = df[df['reportDate'] >= FIRST_USABLE_DATE].reset_index(drop=True)

    if df.empty:
        # print(f"WARNING: [{ticker}] No 10-K/10-Q filings found, returning None for full default fallback")
        return None     

    df = df[['filing_date', 'reportDate', 'form']]
    return df


def get_data_after_company_public(data: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    first_price_date = price_history.index[0]
    data = data[data["fiscalDateEnding"] >= first_price_date]
    return data.reset_index(drop=True)

def encode_signed(a, b):
    both_positive = (a > 0) & (b > 0)
    both_negative = (a < 0) & (b < 0)
    sign_flip_pos = (a > 0) & (b < 0)  # turned profitable
    sign_flip_neg = (a < 0) & (b > 0)  # turned unprofitable

    ratio = np.abs(a) / np.where(np.abs(b) < 1e-10, np.nan, np.abs(b))

    result = np.where(both_positive, ratio,
             np.where(both_negative, ratio,      # same magnitude direction
             np.where(sign_flip_pos, ratio + 2,  # strong positive signal
             np.where(sign_flip_neg, -(ratio + 2), np.nan))))  # strong negative
    return result

def count_past_relative_indicators(df: pd.DataFrame, period: str, past_years: int = 3) -> pd.DataFrame:
    """
    period: 'quarterly' | 'yearly'
    """
    RATIO_COLS = {
        # Margins (0-1 range)
        'grossMargin', 'operatingMargin', 'netProfitMargin', 'ebitdaMargin',
        'fcfMargin', 'ocfToRevenue', 'fcfAfterSbcDilutionCostMargin', 'sbcToRevenue',
        'organicFcfMargin', 
        #'cashReturnToRevenue', 'payoutSustainability',
        # Returns
        'roe', 'roa', 'roic', 'cashKingRatio', 'earningsQuality',
        # Leverage/liquidity (already ratios)
        'currentRatio', 'quickRatio', 'debtToEquity', 'debtToAssets',
        'netDebtToEbitda', 'debtToEbitda', 'fcfToEbitda', 'interestCoverage',
        'equityMultiplier', 'effectiveTaxRate',
        # Capex ratios
        'capexToRevenue', 'capexToOcf', 'capexToDepreciation',
        'reinvestmentRate',
        # Efficiency 
        'grossProfitToAssets', 'revenueToAssets',
        # Valuation multiples, REMOVED AS THEY LEAK PAST PRICE MOVEMENT
        # 'peRatio', 'pbRatio', 'psRatio', 'pfcfRatio', 'pCfoRatio',
        # 'evToEbitda', 'evToFcf', 'evToRevenue', 'evToEbit',
        # Misc ratios
        'dividendPayoutRatio', 'altmanZScore', 'ruleOf40', 'sloanRatio',
    }

    DIFF_COLS = {
        'assetGrowth', 'ocfGrowth', 'fcfGrowth', 'niGrowth',
        'revenueGrowthQoQ', 'epsGrowthQoQ', 'grossMarginDelta',
    }

    SIGNED_RATIO_COLS = {
        'eps', 'bookValuePerShare', 'fcfPerShare', 'ocfPerShare',
        'revenuePerShare', 'fcfAfterSbcDilutionCost', 'ownerEarnings',
        'investedCapital', 'growthCapex', 'dividendYield',
        'organicFcf',
    }

    period_short = 'Q' if period == 'quarterly' else 'Y'
    past_rows = 4 * past_years if period == 'quarterly' else past_years

    ratio_cols = [c for c in df.columns if c in RATIO_COLS]
    diff_cols = [c for c in df.columns if c in DIFF_COLS]
    signed_cols = [c for c in df.columns if c in SIGNED_RATIO_COLS]

    frames = [df]

    for shift_val in range(1, past_rows + 1):
        suffix = f'{period_short}0/{period_short}{shift_val}'
        shifted = df.shift(shift_val)

        if ratio_cols:
            ratio_frame = (df[ratio_cols] / shifted[ratio_cols])
            ratio_frame.columns = [f'{c}{suffix}' for c in ratio_cols]
            frames.append(ratio_frame)
        
        if diff_cols:
            diff_frame = (df[diff_cols] - shifted[diff_cols])
            diff_frame.columns = [f'{c}{suffix}' for c in diff_cols]
            frames.append(diff_frame)

        if signed_cols:
            a = df[signed_cols].values
            b = shifted[signed_cols].values

            both_positive = (a > 0) & (b > 0)
            both_negative = (a < 0) & (b < 0)
            sign_flip_pos = (a > 0) & (b < 0)  # turned profitable
            sign_flip_neg = (a < 0) & (b > 0)  # turned unprofitable

            ratio = np.abs(a) / np.where(np.abs(b) < 1e-10, np.nan, np.abs(b))

            signed_values = np.where(both_positive, ratio,
                            np.where(both_negative, ratio,
                            np.where(sign_flip_pos, ratio + 2,
                            np.where(sign_flip_neg, -(ratio + 2), np.nan))))

            signed_frame = pd.DataFrame(signed_values, index=df.index,
                                        columns=[f'{c}{suffix}' for c in signed_cols])
            frames.append(signed_frame)

    
    result = pd.concat(frames, axis=1)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result

def count_future_diff_pct(prices: pd.Series, period: str, years: int = 1) -> pd.Series:

    shift_val = -years * 4 if period == 'quarterly' else -years
    prices_future = prices.shift(shift_val)
    
    prices_diff = prices_future - prices
    prices_diff_pct = (prices_diff / prices) * 100

    return prices_diff_pct

def count_stock_advantage_over_market(stock_prices_diff_pct: pd.Series, market_prices_diff_pct: pd.Series) -> pd.Series:
    return stock_prices_diff_pct - market_prices_diff_pct

def trim_to_continuous_range(dates: pd.Series, period: str) -> tuple[int, int]:
    max_diff = 95 if period == 'quarterly' else 390
    gaps = dates.diff().dt.days
    bad_positions = gaps[gaps > max_diff].index.tolist()

    if not bad_positions:
        return 0, len(dates) - 1

    head_range = (0, bad_positions[0] - 1)
    tail_range = (bad_positions[-1], len(dates) - 1)

    head_len = head_range[1] - head_range[0] + 1
    tail_len = tail_range[1] - tail_range[0] + 1

    return head_range if head_len >= tail_len else tail_range


def convert_values_to_usd(df: pd.DataFrame) -> pd.DataFrame | None:
    currencies = [c for c in df['reportedCurrency'].unique().tolist() 
                  if c not in (None, 'None') and pd.notna(c) and c != 'USD']
    if "USD" in currencies:
        currencies.remove("USD")

    NON_NUMERIC = {'fiscalDateEnding', 'reportedCurrency'}
    numeric_cols = [c for c in df.columns if c not in NON_NUMERIC]

    for foreign_currency in currencies:
        try:
            currency_ratio_history = get_price_history(f"{foreign_currency}USD=X")
            ratio = lookup_prices_vectorized(currency_ratio_history, df['fiscalDateEnding'])

            if ratio is None:
                print(f"ERROR Collecting {foreign_currency}USD ratio failed")
                return None

        except Exception as e:
            print(f"ERROR Collecting {foreign_currency}USD ratio failed")
            return None

        df[numeric_cols] = df[numeric_cols].astype(float)

        mask = df['reportedCurrency'] == foreign_currency
        df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].multiply(ratio[mask], axis=0)
    return df
