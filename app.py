import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import json
from datetime import datetime, timedelta, time as dtime
import pytz

# Configuração da página
st.set_page_config(
    page_title="Radar B3: Robô Trader, Carteiras & Simulador",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2a2e39;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
    .status-aberto {
        color: #00E676;
        font-weight: bold;
    }
    .status-fechado {
        color: #EF5350;
        font-weight: bold;
    }
    .status-leilao {
        color: #FFA726;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Checagem de Horário Oficial da Bolsa B3 (Horário de Brasília / UTC-3)
# -----------------------------------------------------------------------------
def obter_status_mercado_b3():
    """Verifica se a B3 está em horário de pregão oficial."""
    fuso_sp = pytz.timezone("America/Sao_Paulo")
    agora_sp = datetime.now(fuso_sp)
    dia_semana = agora_sp.weekday()  # 0=Segunda, 4=Sexta, 5=Sábado, 6=Domingo
    hora_atual = agora_sp.time()

    abertura_leilao = dtime(9, 45)
    abertura_pregao = dtime(10, 0)
    fechamento_pregao = dtime(17, 0)
    fechamento_after = dtime(17, 55)

    if dia_semana >= 5:
        return "🔴 MERCADO FECHADO (Fim de Semana)", "Fora do Horário", agora_sp
    elif hora_atual < abertura_leilao:
        return "🔴 MERCADO FECHADO (Antes do Pregão)", "Abre às 10:00", agora_sp
    elif abertura_leilao <= hora_atual < abertura_pregao:
        return "🟡 LEILÃO DE PRÉ-ABERTURA (09:45 - 10:00)", "Abertura iminente", agora_sp
    elif abertura_pregao <= hora_atual <= fechamento_pregao:
        return "🟢 PREGÃO REGULAR ABERTO (10:00 - 17:00)", "Ordens a Mercado Ativas", agora_sp
    elif fechamento_pregao < hora_atual <= fechamento_after:
        return "🟡 LEILÃO DE FECHAMENTO / AFTER (17:00 - 17:55)", "Ajustes de Posição", agora_sp
    else:
        return "🔴 MERCADO FECHADO (Após Pregão)", "Encerramento do Dia", agora_sp

# -----------------------------------------------------------------------------
# 2. Indicadores Macroeconômicos (Banco Central do Brasil / SGS)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def obter_dados_macro():
    """Busca dados oficiais da Selic no SGS do Banco Central do Brasil."""
    dados = {
        "selic": 13.65,
        "cdi": 13.55,
        "data": datetime.now().strftime("%d/%m/%Y")
    }
    try:
        url_selic = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados/ultimos/1?formato=json"
        res_selic = requests.get(url_selic, timeout=5).json()
        if res_selic:
            selic_val = float(res_selic[0]["valor"])
            dados["selic"] = selic_val
            dados["cdi"] = round(selic_val - 0.10, 2)
            dados["data"] = res_selic[0]["data"]
    except Exception:
        pass
    return dados

# -----------------------------------------------------------------------------
# 3. Coleta de Todos os Tickers da B3 em Tempo Real
# -----------------------------------------------------------------------------
@st.cache_data(ttl=120)
def carregar_todos_tickers_b3():
    """Coleta os ativos listados na B3 em tempo real."""
    todos_ativos = []
    page = 1
    
    while True:
        try:
            url = f"https://brapi.dev/api/quote/list?limit=1000&page={page}"
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                break
            dados = res.json()
            stocks = dados.get("stocks", [])
            if not stocks:
                break
            todos_ativos.extend(stocks)
            if not dados.get("hasNextPage", False):
                break
            page += 1
        except Exception:
            break

    if not todos_ativos:
        return pd.DataFrame()

    df = pd.DataFrame(todos_ativos)
    df["stock"] = df["stock"].fillna("").astype(str).str.upper()
    df["name"] = df["name"].fillna(df["stock"]).astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df["change"] = pd.to_numeric(df["change"], errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0.0)
    df["sector"] = df["sector"].fillna("Outros").astype(str)

    df["vol_financeiro_mi"] = (df["close"] * df["volume"]) / 1_000_000

    def mapear_tipo(t, ticker):
        t_str = str(t).lower()
        if t_str == "fund" or any(num in ticker for num in ["11", "11B"]):
            return "FII / Fundo"
        elif t_str == "bdr" or any(num in ticker for num in ["34", "35", "32"]):
            return "BDR"
        else:
            return "Ação"

    df["tipo_desc"] = [mapear_tipo(t, s) for t, s in zip(df.get("type", []), df["stock"])]

    df = df.rename(columns={
        "stock": "Ticker",
        "name": "Nome / Razão Social",
        "close": "Preço Atual (R$)",
        "change": "Variação Dia (%)",
        "volume": "Volume Cotas/Ações",
        "vol_financeiro_mi": "Volume Financeiro (R$ Mi)",
        "tipo_desc": "Classe",
        "sector": "Setor"
    })

    return df

# -----------------------------------------------------------------------------
# 4. Indicadores Técnicos Avançados (Arsenal do Corretor B3)
# -----------------------------------------------------------------------------
import yfinance as yf

def calcular_ema(serie, periodo):
    """Média Móvel Exponencial."""
    return serie.ewm(span=periodo, adjust=False).mean()

def calcular_sma(serie, periodo):
    """Média Móvel Simples."""
    return serie.rolling(window=periodo).mean()

def calcular_rsi(serie, periodo=14):
    """IFR / RSI (Índice de Força Relativa)."""
    delta = serie.diff()
    ganho = delta.clip(lower=0).rolling(window=periodo).mean()
    perda = (-delta.clip(upper=0)).rolling(window=periodo).mean()
    rs = ganho / (perda + 1e-10)
    return 100 - (100 / (1 + rs))

def calcular_macd(serie, rapida=12, lenta=26, sinal=9):
    """MACD (Moving Average Convergence Divergence)."""
    ema_rapida = calcular_ema(serie, rapida)
    ema_lenta = calcular_ema(serie, lenta)
    macd_line = ema_rapida - ema_lenta
    signal_line = calcular_ema(macd_line, sinal)
    histograma = macd_line - signal_line
    return macd_line, signal_line, histograma

def calcular_bollinger(serie, periodo=20, desvios=2.0):
    """Bandas de Bollinger."""
    media = calcular_sma(serie, periodo)
    std = serie.rolling(window=periodo).std()
    banda_sup = media + (desvios * std)
    banda_inf = media - (desvios * std)
    percentual_b = (serie - banda_inf) / (banda_sup - banda_inf + 1e-10)
    return banda_sup, media, banda_inf, percentual_b

def calcular_adx(high, low, close, periodo=14):
    """ADX (Average Directional Index) - Força da tendência."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=periodo).mean()
    plus_di = 100 * (plus_dm.rolling(window=periodo).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=periodo).mean() / (atr + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(window=periodo).mean()
    return adx, plus_di, minus_di, atr

def calcular_obv(close, volume):
    """OBV (On Balance Volume) - Fluxo institucional."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv

def calcular_mfi(high, low, close, volume, periodo=14):
    """MFI (Money Flow Index) - IFR ponderado por volume."""
    preco_tipico = (high + low + close) / 3
    money_flow_raw = preco_tipico * volume
    positivo = money_flow_raw.where(preco_tipico > preco_tipico.shift(1), 0.0).rolling(periodo).sum()
    negativo = money_flow_raw.where(preco_tipico < preco_tipico.shift(1), 0.0).rolling(periodo).sum()
    ratio = positivo / (negativo + 1e-10)
    return 100 - (100 / (1 + ratio))

def calcular_estocastico(high, low, close, k_periodo=14, d_periodo=3):
    """Estocástico %K e %D."""
    low_min = low.rolling(window=k_periodo).min()
    high_max = high.rolling(window=k_periodo).max()
    k = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
    d = k.rolling(window=d_periodo).mean()
    return k, d

def calcular_vwap(high, low, close, volume):
    """VWAP (Volume Weighted Average Price) - Preço médio institucional."""
    preco_tipico = (high + low + close) / 3
    vwap_acum = (preco_tipico * volume).cumsum() / (volume.cumsum() + 1e-10)
    return vwap_acum

def calcular_atr(high, low, close, periodo=14):
    """ATR (Average True Range) para sizing e stops dinâmicos."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=periodo).mean()

# -----------------------------------------------------------------------------
# 5. Motor de Simulação do Robô Trader B3 (Engine Profissional)
# -----------------------------------------------------------------------------
def gerar_score_confluencia(serie_close, serie_high, serie_low, serie_volume, idx):
    """
    Sistema de Confluência Multi-Indicador (Profissional B3).
    Cada indicador técnico vota de -2 a +2 pontos. O score final
    determina a intensidade e a direção do sinal.
    Retorna (score_total, detalhes_dict).
    """
    close = serie_close.iloc[:idx+1]
    high = serie_high.iloc[:idx+1]
    low = serie_low.iloc[:idx+1]
    vol = serie_volume.iloc[:idx+1]
    
    if len(close) < 30:
        return 0, {}

    score = 0
    detalhes = {}
    
    # 1. MACD (Convergência/Divergência)
    macd_l, macd_s, macd_h = calcular_macd(close)
    if len(macd_h) >= 2:
        macd_val = float(macd_h.iloc[-1])
        macd_prev = float(macd_h.iloc[-2])
        if macd_val > 0 and macd_val > macd_prev:
            score += 2; detalhes["MACD"] = f"+2 (Histograma positivo e crescente: {macd_val:.3f})"
        elif macd_val > 0:
            score += 1; detalhes["MACD"] = f"+1 (Histograma positivo: {macd_val:.3f})"
        elif macd_val < 0 and macd_val < macd_prev:
            score -= 2; detalhes["MACD"] = f"-2 (Histograma negativo e caindo: {macd_val:.3f})"
        else:
            score -= 1; detalhes["MACD"] = f"-1 (Histograma negativo: {macd_val:.3f})"

    # 2. IFR / RSI
    rsi = calcular_rsi(close, 14)
    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50
    if rsi_val < 30:
        score += 2; detalhes["IFR/RSI"] = f"+2 (Sobrevendido: {rsi_val:.1f})"
    elif rsi_val < 40:
        score += 1; detalhes["IFR/RSI"] = f"+1 (Região de compra: {rsi_val:.1f})"
    elif rsi_val > 70:
        score -= 2; detalhes["IFR/RSI"] = f"-2 (Sobrecomprado: {rsi_val:.1f})"
    elif rsi_val > 60:
        score -= 1; detalhes["IFR/RSI"] = f"-1 (Esticado: {rsi_val:.1f})"
    else:
        detalhes["IFR/RSI"] = f"0 (Neutro: {rsi_val:.1f})"

    # 3. Bandas de Bollinger
    bb_sup, bb_med, bb_inf, pct_b = calcular_bollinger(close)
    pct_b_val = float(pct_b.iloc[-1]) if not pct_b.empty else 0.5
    if pct_b_val < 0.05:
        score += 2; detalhes["Bollinger"] = f"+2 (Abaixo da banda inferior: %B={pct_b_val:.2f})"
    elif pct_b_val < 0.20:
        score += 1; detalhes["Bollinger"] = f"+1 (Próximo da banda inferior: %B={pct_b_val:.2f})"
    elif pct_b_val > 0.95:
        score -= 2; detalhes["Bollinger"] = f"-2 (Acima da banda superior: %B={pct_b_val:.2f})"
    elif pct_b_val > 0.80:
        score -= 1; detalhes["Bollinger"] = f"-1 (Próximo da banda superior: %B={pct_b_val:.2f})"
    else:
        detalhes["Bollinger"] = f"0 (Zona neutra: %B={pct_b_val:.2f})"

    # 4. ADX (Força da Tendência) + DI
    adx, plus_di, minus_di, atr = calcular_adx(high, low, close)
    adx_val = float(adx.iloc[-1]) if not adx.empty else 0
    pdi = float(plus_di.iloc[-1]) if not plus_di.empty else 0
    mdi = float(minus_di.iloc[-1]) if not minus_di.empty else 0
    if adx_val > 25 and pdi > mdi:
        score += 2; detalhes["ADX"] = f"+2 (Tendência forte de alta: ADX={adx_val:.1f}, +DI>{'-DI'})"
    elif adx_val > 20 and pdi > mdi:
        score += 1; detalhes["ADX"] = f"+1 (Tendência moderada de alta: ADX={adx_val:.1f})"
    elif adx_val > 25 and mdi > pdi:
        score -= 2; detalhes["ADX"] = f"-2 (Tendência forte de baixa: ADX={adx_val:.1f})"
    elif adx_val > 20 and mdi > pdi:
        score -= 1; detalhes["ADX"] = f"-1 (Tendência moderada de baixa: ADX={adx_val:.1f})"
    else:
        detalhes["ADX"] = f"0 (Sem tendência definida: ADX={adx_val:.1f})"

    # 5. OBV (On Balance Volume) - Fluxo institucional
    obv = calcular_obv(close, vol)
    if len(obv) >= 10:
        obv_ema = calcular_ema(obv, 10)
        obv_val = float(obv.iloc[-1])
        obv_ema_val = float(obv_ema.iloc[-1])
        if obv_val > obv_ema_val:
            score += 1; detalhes["OBV"] = f"+1 (Fluxo comprador acima da média)"
        else:
            score -= 1; detalhes["OBV"] = f"-1 (Fluxo vendedor abaixo da média)"

    # 6. MFI (Money Flow Index)
    mfi = calcular_mfi(high, low, close, vol)
    mfi_val = float(mfi.iloc[-1]) if not mfi.empty else 50
    if mfi_val < 20:
        score += 2; detalhes["MFI"] = f"+2 (Pressão compradora extrema: MFI={mfi_val:.1f})"
    elif mfi_val < 35:
        score += 1; detalhes["MFI"] = f"+1 (Zona de acumulação: MFI={mfi_val:.1f})"
    elif mfi_val > 80:
        score -= 2; detalhes["MFI"] = f"-2 (Pressão vendedora extrema: MFI={mfi_val:.1f})"
    elif mfi_val > 65:
        score -= 1; detalhes["MFI"] = f"-1 (Zona de distribuição: MFI={mfi_val:.1f})"
    else:
        detalhes["MFI"] = f"0 (Equilíbrio: MFI={mfi_val:.1f})"

    # 7. Estocástico %K / %D
    k, d = calcular_estocastico(high, low, close)
    k_val = float(k.iloc[-1]) if not k.empty else 50
    d_val = float(d.iloc[-1]) if not d.empty else 50
    if k_val < 20 and k_val > d_val:
        score += 2; detalhes["Estocástico"] = f"+2 (Sobrevendido com cruzamento: %K={k_val:.1f})"
    elif k_val < 30:
        score += 1; detalhes["Estocástico"] = f"+1 (Zona sobrevendida: %K={k_val:.1f})"
    elif k_val > 80 and k_val < d_val:
        score -= 2; detalhes["Estocástico"] = f"-2 (Sobrecomprado com cruzamento: %K={k_val:.1f})"
    elif k_val > 70:
        score -= 1; detalhes["Estocástico"] = f"-1 (Zona sobrecomprada: %K={k_val:.1f})"
    else:
        detalhes["Estocástico"] = f"0 (Neutro: %K={k_val:.1f})"

    # 8. Médias Móveis (EMA 9 x EMA 21 x EMA 50)
    ema9 = calcular_ema(close, 9)
    ema21 = calcular_ema(close, 21)
    ema9v = float(ema9.iloc[-1])
    ema21v = float(ema21.iloc[-1])
    preco = float(close.iloc[-1])
    if preco > ema9v > ema21v:
        score += 2; detalhes["EMAs"] = f"+2 (Preço > EMA9 > EMA21 — alinhamento altista)"
    elif ema9v > ema21v:
        score += 1; detalhes["EMAs"] = f"+1 (EMA9 > EMA21 — tendência de alta)"
    elif preco < ema9v < ema21v:
        score -= 2; detalhes["EMAs"] = f"-2 (Preço < EMA9 < EMA21 — alinhamento baixista)"
    elif ema9v < ema21v:
        score -= 1; detalhes["EMAs"] = f"-1 (EMA9 < EMA21 — tendência de baixa)"
    else:
        detalhes["EMAs"] = f"0 (Indefinido)"

    # 9. Gap de Abertura (padrão comum na B3 por conta de ADRs e mercado externo)
    if len(close) >= 2:
        gap_pct = ((float(close.iloc[-1]) / float(close.iloc[-2])) - 1) * 100
        if gap_pct > 2.0:
            score += 1; detalhes["Gap"] = f"+1 (Gap de alta: +{gap_pct:.2f}%)"
        elif gap_pct < -2.0:
            score -= 1; detalhes["Gap"] = f"-1 (Gap de baixa: {gap_pct:.2f}%)"
        else:
            detalhes["Gap"] = f"0 (Sem gap significativo: {gap_pct:+.2f}%)"

    return score, detalhes


@st.cache_data(ttl=300)
def executar_robo_trader(capital_inicial, dt_inicio, dt_fim, estrategia, stop_loss_pct, take_profit_pct, universo_tickers, trailing_stop_ativo, risco_por_trade_pct):
    """
    Motor Profissional de Trading Algorítmico da B3.
    
    Técnicas implementadas:
    - Sistema de Confluência Multi-Indicador (MACD, RSI, Bollinger, ADX, OBV, MFI, Estocástico, EMAs, Gap)
    - Trailing Stop dinâmico baseado em ATR
    - Position Sizing com Kelly Criterion simplificado
    - Emolumentos e taxas oficiais da B3 (0.03%)
    - Gestão de risco por operação com teto de alocação
    """
    # Baixamos 90 dias antes de dt_inicio APENAS para warm-up dos indicadores (MACD, RSI, ADX, etc.)
    # MAS nenhuma ordem é executada antes da data real estipulada!
    dt_warmup_str = (dt_inicio - timedelta(days=90)).strftime("%Y-%m-%d")
    dt_fim_ajustada = (dt_fim + timedelta(days=2)).strftime("%Y-%m-%d")
    
    tickers_sa = [f"{t}.SA" if not t.endswith(".SA") else t for t in universo_tickers]
    
    try:
        dados_mercado = yf.download(tickers_sa, start=dt_warmup_str, end=dt_fim_ajustada, threads=True, progress=False)
    except Exception:
        return None

    if dados_mercado.empty:
        return None

    close_df = dados_mercado["Close"]
    high_df = dados_mercado["High"]
    low_df = dados_mercado["Low"]
    volume_df = dados_mercado["Volume"]
    datas_pregao = close_df.index

    if len(datas_pregao) < 5:
        return None

    caixa = float(capital_inicial)
    custo_emolumentos_b3 = 0.0003  # 0.03% por transação (emolumentos + liquidação CBLC)
    max_posicoes = 5
    teto_alocacao_pct = risco_por_trade_pct / 100  # % do capital por posição
    
    posicoes = {}
    livro_ordens = []
    historico_patrimonio = []
    indicadores_log = []  # log dos indicadores técnicos por dia

    # Histórico de trades para cálculo Kelly
    historico_resultados = []

    for i in range(len(datas_pregao)):
        dia = datas_pregao[i]
        dia_date = dia.date() if hasattr(dia, "date") else pd.to_datetime(dia).date()
        
        # REGRA ABSOLUTA DE OPERAÇÃO:
        # O robô NÃO opera antes da data de início estipulada pelo usuário!
        if dia_date < dt_inicio:
            continue
            
        if dia_date > dt_fim:
            break

        dia_str = dia.strftime("%d/%m/%Y")
        patrimonio_corrente = caixa
        for t, pos in posicoes.items():
            if t in close_df:
                p = float(close_df[t].iloc[i])
                if not np.isnan(p):
                    patrimonio_corrente += pos["qtd"] * p
        
        # =====================================================================
        # FASE 1: GESTÃO DE POSIÇÕES ABERTAS (Verificação de Saídas)
        # =====================================================================
        tickers_em_custodia = list(posicoes.keys())
        for t in tickers_em_custodia:
            if t not in close_df:
                continue
            preco_fech = float(close_df[t].iloc[i])
            if np.isnan(preco_fech) or preco_fech <= 0:
                continue

            pos = posicoes[t]
            retorno_pos = (preco_fech / pos["preco_entrada"]) - 1.0

            gatilho_saida = False
            motivo_saida = ""
            tipo_saida = ""

            # Trailing Stop Dinâmico (baseado em ATR)
            if trailing_stop_ativo and t in high_df and t in low_df:
                atr_serie = calcular_atr(high_df[t].iloc[:i+1], low_df[t].iloc[:i+1], close_df[t].iloc[:i+1])
                atr_val = float(atr_serie.iloc[-1]) if not atr_serie.empty and not np.isnan(atr_serie.iloc[-1]) else 0
                
                # Atualizar preço máximo atingido
                if preco_fech > pos.get("preco_max", pos["preco_entrada"]):
                    posicoes[t]["preco_max"] = preco_fech
                
                preco_max = pos.get("preco_max", pos["preco_entrada"])
                trailing_level = preco_max - (2.5 * atr_val) if atr_val > 0 else preco_max * (1 - stop_loss_pct/100)
                
                if preco_fech <= trailing_level and retorno_pos > 0.005:
                    gatilho_saida = True
                    motivo_saida = f"📉 TRAILING STOP (ATR) — Máximo: R${preco_max:.2f}, Nível: R${trailing_level:.2f}"
                    tipo_saida = "TRAILING"

            # Stop Loss Fixo
            if not gatilho_saida and retorno_pos <= -(stop_loss_pct / 100):
                gatilho_saida = True
                motivo_saida = f"🛑 STOP LOSS (-{stop_loss_pct:.1f}%) — Entrada: R${pos['preco_entrada']:.2f}"
                tipo_saida = "STOP"
            
            # Take Profit Fixo
            if not gatilho_saida and retorno_pos >= (take_profit_pct / 100):
                gatilho_saida = True
                motivo_saida = f"🎯 TAKE PROFIT (+{take_profit_pct:.1f}%) — Lucro realizado"
                tipo_saida = "TAKE"

            # Sinal técnico de reversão (confluência negativa)
            if not gatilho_saida and i >= 30:
                if estrategia in ["Confluência Multi-Indicador (Profissional)", "MACD + Bollinger (Volatilidade)", "Momentum com ADX (Força de Tendência)"]:
                    if t in high_df and t in low_df and t in volume_df:
                        score_atual, _ = gerar_score_confluencia(close_df[t], high_df[t], low_df[t], volume_df[t], i)
                        if score_atual <= -6:
                            gatilho_saida = True
                            motivo_saida = f"🔄 REVERSÃO TÉCNICA — Score Confluência: {score_atual} (crítico)"
                            tipo_saida = "REVERSAL"
                elif estrategia == "Cruzamento de Médias (EMA 9 x EMA 21)":
                    serie = close_df[t].iloc[:i+1].dropna()
                    if len(serie) >= 21:
                        ema9 = calcular_ema(serie, 9).iloc[-1]
                        ema21 = calcular_ema(serie, 21).iloc[-1]
                        if ema9 < ema21:
                            gatilho_saida = True
                            motivo_saida = "🔄 REVERSÃO — EMA 9 cruzou abaixo da EMA 21"
                            tipo_saida = "REVERSAL"

            if gatilho_saida:
                valor_bruto_venda = pos["qtd"] * preco_fech
                taxa_venda = valor_bruto_venda * custo_emolumentos_b3
                valor_liquido = valor_bruto_venda - taxa_venda
                custo_compra_total = (pos["qtd"] * pos["preco_entrada"]) + pos["taxas"]
                resultado = valor_liquido - custo_compra_total
                ret_pct = (resultado / custo_compra_total) * 100

                caixa += valor_liquido
                historico_resultados.append(resultado)

                livro_ordens.append({
                    "Data Pregão": dia_str,
                    "Horário B3": "16:45:00",
                    "Ticker": t.replace(".SA", ""),
                    "Operação": "VENDA",
                    "Qtd": pos["qtd"],
                    "Preço Execução (R$)": round(preco_fech, 2),
                    "Taxas B3 (R$)": round(taxa_venda, 2),
                    "Resultado Trade (R$)": round(resultado, 2),
                    "Retorno (%)": round(ret_pct, 2),
                    "Motivo Execução": motivo_saida
                })
                del posicoes[t]

        # =====================================================================
        # FASE 2: BUSCA DE NOVAS OPORTUNIDADES DE COMPRA (Pregão 10:00 às 17:00)
        # =====================================================================
        if len(posicoes) < max_posicoes and caixa >= 1000.0 and i >= 30:
            
            candidatos = []
            for t in tickers_sa:
                if t in posicoes or t not in close_df or t not in high_df or t not in low_df or t not in volume_df:
                    continue

                serie_c = close_df[t].iloc[:i+1].dropna()
                if len(serie_c) < 30:
                    continue

                preco_atual = float(serie_c.iloc[-1])
                if np.isnan(preco_atual) or preco_atual <= 0:
                    continue

                sinal_compra = False
                motivo_compra = ""
                score_total = 0

                if estrategia == "Confluência Multi-Indicador (Profissional)":
                    score_total, det = gerar_score_confluencia(close_df[t], high_df[t], low_df[t], volume_df[t], i)
                    if score_total >= 5:
                        sinal_compra = True
                        motivo_compra = f"✅ Confluência Forte ({score_total} pts): " + "; ".join([f"{k}={v.split('(')[0].strip()}" for k, v in det.items() if v.startswith("+")])
                
                elif estrategia == "MACD + Bollinger (Volatilidade)":
                    macd_l, macd_s, macd_h = calcular_macd(serie_c)
                    _, _, _, pct_b = calcular_bollinger(serie_c)
                    if len(macd_h) >= 2 and len(pct_b) >= 1:
                        macd_val = float(macd_h.iloc[-1])
                        macd_prev = float(macd_h.iloc[-2])
                        pctb_val = float(pct_b.iloc[-1])
                        if macd_val > 0 and macd_val > macd_prev and pctb_val < 0.5:
                            sinal_compra = True
                            score_total = 6
                            motivo_compra = f"✅ MACD acelerando (+{macd_val:.3f}) com Bollinger favorável (%B={pctb_val:.2f})"

                elif estrategia == "Momentum com ADX (Força de Tendência)":
                    adx, pdi, mdi, atr = calcular_adx(high_df[t].iloc[:i+1], low_df[t].iloc[:i+1], serie_c)
                    rsi = calcular_rsi(serie_c)
                    adx_val = float(adx.iloc[-1]) if not adx.empty else 0
                    pdi_val = float(pdi.iloc[-1]) if not pdi.empty else 0
                    mdi_val = float(mdi.iloc[-1]) if not mdi.empty else 0
                    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50
                    if adx_val > 25 and pdi_val > mdi_val and rsi_val < 65:
                        sinal_compra = True
                        score_total = 7
                        motivo_compra = f"✅ Tendência forte (ADX={adx_val:.1f}), +DI dominante, RSI={rsi_val:.1f}"

                elif estrategia == "IFR + Estocástico (Reversão à Média)":
                    rsi = calcular_rsi(serie_c)
                    k, d = calcular_estocastico(high_df[t].iloc[:i+1], low_df[t].iloc[:i+1], serie_c)
                    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50
                    k_val = float(k.iloc[-1]) if not k.empty else 50
                    d_val = float(d.iloc[-1]) if not d.empty else 50
                    if rsi_val < 35 and k_val < 25 and k_val > d_val:
                        sinal_compra = True
                        score_total = 8
                        motivo_compra = f"✅ RSI sobrevendido ({rsi_val:.1f}) + Estocástico cruzando (%K={k_val:.1f} > %D={d_val:.1f})"

                elif estrategia == "VWAP + OBV (Fluxo Institucional)":
                    vwap = calcular_vwap(high_df[t].iloc[:i+1], low_df[t].iloc[:i+1], serie_c, volume_df[t].iloc[:i+1])
                    obv = calcular_obv(serie_c, volume_df[t].iloc[:i+1])
                    vwap_val = float(vwap.iloc[-1]) if not vwap.empty else preco_atual
                    obv_ema = calcular_ema(obv, 10) if len(obv) >= 10 else obv
                    obv_val = float(obv.iloc[-1]) if not obv.empty else 0
                    obv_ema_val = float(obv_ema.iloc[-1]) if not obv_ema.empty else 0
                    if preco_atual > vwap_val and obv_val > obv_ema_val:
                        sinal_compra = True
                        score_total = 6
                        motivo_compra = f"✅ Preço acima do VWAP (R${vwap_val:.2f}) com OBV institucional comprador"

                elif estrategia == "Cruzamento de Médias (EMA 9 x EMA 21)":
                    ema9 = calcular_ema(serie_c, 9).iloc[-1]
                    ema21 = calcular_ema(serie_c, 21).iloc[-1]
                    if ema9 > ema21 and float(serie_c.iloc[-1]) > float(serie_c.iloc[-2]):
                        sinal_compra = True
                        score_total = 5
                        motivo_compra = f"✅ Cruzamento EMA9 ({ema9:.2f}) > EMA21 ({ema21:.2f}) com candle de alta"

                elif estrategia == "IFR / RSI Sobrevendido":
                    rsi = calcular_rsi(serie_c, 14)
                    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50
                    if rsi_val < 35:
                        sinal_compra = True
                        score_total = 5
                        motivo_compra = f"✅ IFR Sobrevendido (RSI={rsi_val:.1f} < 35)"

                if sinal_compra:
                    candidatos.append((t, score_total, preco_atual, motivo_compra))

            # Ordenar candidatos pelo score (melhor primeiro)
            candidatos.sort(key=lambda x: x[1], reverse=True)

            for t, score_c, preco_atual, motivo_compra in candidatos:
                if len(posicoes) >= max_posicoes or caixa < 1000:
                    break

                # Position Sizing: Kelly Criterion simplificado ou percentual fixo
                if len(historico_resultados) >= 5:
                    wins = [r for r in historico_resultados if r > 0]
                    losses = [abs(r) for r in historico_resultados if r < 0]
                    win_rate = len(wins) / len(historico_resultados) if historico_resultados else 0.5
                    avg_win = np.mean(wins) if wins else 1
                    avg_loss = np.mean(losses) if losses else 1
                    kelly_frac = (win_rate * avg_win - (1 - win_rate) * avg_loss) / (avg_win + 1e-10)
                    kelly_frac = max(0.05, min(kelly_frac, teto_alocacao_pct))
                else:
                    kelly_frac = teto_alocacao_pct

                capital_alocar = min(caixa * 0.95, patrimonio_corrente * kelly_frac)
                # Regra B3: Lote Padrão de 100 em 100 ações (não fracionado)
                lote_padrao = 100
                num_lotes = int(capital_alocar // (preco_atual * lote_padrao))
                qtd = num_lotes * lote_padrao
                
                if qtd >= 100:
                    custo_bruto = qtd * preco_atual
                    taxa_compra = custo_bruto * custo_emolumentos_b3
                    custo_total = custo_bruto + taxa_compra

                    if caixa >= custo_total:
                        caixa -= custo_total
                        posicoes[t] = {
                            "qtd": qtd,
                            "preco_entrada": preco_atual,
                            "data_entrada": dia_str,
                            "taxas": taxa_compra,
                            "preco_max": preco_atual,
                            "score_entrada": score_c
                        }
                        livro_ordens.append({
                            "Data Pregão": dia_str,
                            "Horário B3": "10:15:30",
                            "Ticker": t.replace(".SA", ""),
                            "Operação": "COMPRA",
                            "Qtd": qtd,
                            "Preço Execução (R$)": round(preco_atual, 2),
                            "Taxas B3 (R$)": round(taxa_compra, 2),
                            "Resultado Trade (R$)": 0.0,
                            "Retorno (%)": 0.0,
                            "Motivo Execução": motivo_compra
                        })

        # =====================================================================
        # FASE 3: CONSOLIDAÇÃO DE PATRIMÔNIO (Fechamento 17:00)
        # =====================================================================
        valor_custodia = 0.0
        for t, pos in posicoes.items():
            if t in close_df:
                p = float(close_df[t].iloc[i])
                if not np.isnan(p):
                    valor_custodia += pos["qtd"] * p

        historico_patrimonio.append({
            "Data": dia,
            "Caixa (R$)": round(caixa, 2),
            "Ações em Custódia (R$)": round(valor_custodia, 2),
            "Patrimônio Total (R$)": round(caixa + valor_custodia, 2)
        })

    # Tabela de posições abertas restantes
    posicoes_finais = []
    for t, pos in posicoes.items():
        if t in close_df:
            p_ultimo = float(close_df[t].iloc[-1])
            valor_atual = pos["qtd"] * p_ultimo
            custo_compra = pos["qtd"] * pos["preco_entrada"]
            pnl = valor_atual - custo_compra
            pnl_pct = (pnl / custo_compra) * 100
            posicoes_finais.append({
                "Ticker": t.replace(".SA", ""),
                "Data Entrada": pos["data_entrada"],
                "Qtd": pos["qtd"],
                "Preço Compra (R$)": round(pos["preco_entrada"], 2),
                "Preço Atual (R$)": round(p_ultimo, 2),
                "Valor em Custódia (R$)": round(valor_atual, 2),
                "Resultado Aberto (R$)": round(pnl, 2),
                "Retorno (%)": round(pnl_pct, 2),
                "Score Entrada": pos.get("score_entrada", 0)
            })

    if not historico_patrimonio:
        historico_patrimonio.append({
            "Data": pd.to_datetime(dt_inicio),
            "Caixa (R$)": round(caixa, 2),
            "Ações em Custódia (R$)": 0.0,
            "Patrimônio Total (R$)": round(caixa, 2)
        })

    return {
        "historico": pd.DataFrame(historico_patrimonio),
        "ordens": pd.DataFrame(livro_ordens),
        "posicoes_abertas": pd.DataFrame(posicoes_finais),
        "caixa_final": caixa,
        "patrimonio_final": historico_patrimonio[-1]["Patrimônio Total (R$)"] if historico_patrimonio else capital_inicial
    }

# -----------------------------------------------------------------------------
# 5. Gerenciamento de Estado Persistente do Robô Trader (Ciclos Programados)
# -----------------------------------------------------------------------------
ARQUIVO_ESTADO_ROBO = "robo_state.json"

def carregar_estado_robo():
    if os.path.exists(ARQUIVO_ESTADO_ROBO):
        try:
            with open(ARQUIVO_ESTADO_ROBO, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def salvar_estado_robo(estado):
    try:
        with open(ARQUIVO_ESTADO_ROBO, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass
    salvar_no_banco_historico(estado)

def resetar_ciclo_robo():
    if os.path.exists(ARQUIVO_ESTADO_ROBO):
        try:
            os.remove(ARQUIVO_ESTADO_ROBO)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Banco de Histórico Permanente de Ciclos (Auditoria e Análise Futura)
# -----------------------------------------------------------------------------
ARQUIVO_BANCO_HISTORICO = "historico_banco_robo.json"

def carregar_banco_historico():
    if os.path.exists(ARQUIVO_BANCO_HISTORICO):
        try:
            with open(ARQUIVO_BANCO_HISTORICO, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def salvar_no_banco_historico(estado):
    if not estado:
        return
    banco = carregar_banco_historico()
    id_ciclo = estado.get("id_ciclo")
    if not id_ciclo:
        id_ciclo = f"CICLO_{estado.get('data_inicio', '').replace('-', '')}_{datetime.now().strftime('%H%M%S')}"
        estado["id_ciclo"] = id_ciclo
    
    cap_ini = float(estado.get("capital_inicial", 0))
    patr_fim = float(estado.get("patrimonio_final", cap_ini))
    lucro = patr_fim - cap_ini
    ret_pct = (lucro / cap_ini * 100) if cap_ini > 0 else 0.0
    
    ordens = estado.get("ordens", [])
    vendas = [o for o in ordens if o.get("Operação") == "VENDA"]
    total_trades = len(vendas)
    venc = len([o for o in vendas if o.get("Resultado Trade (R$)", 0) > 0])
    win_rate = (venc / total_trades * 100) if total_trades > 0 else 0.0
    
    ganhos = sum([o.get("Resultado Trade (R$)", 0) for o in vendas if o.get("Resultado Trade (R$)", 0) > 0])
    perdas = abs(sum([o.get("Resultado Trade (R$)", 0) for o in vendas if o.get("Resultado Trade (R$)", 0) < 0]))
    pf = (ganhos / perdas) if perdas > 0 else (ganhos if ganhos > 0 else 1.0)
    
    registro = {
        "id_ciclo": id_ciclo,
        "data_registro": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "data_inicio": estado.get("data_inicio"),
        "data_fim": estado.get("data_fim"),
        "estrategia": estado.get("estrategia"),
        "status": estado.get("status"),
        "capital_inicial": cap_ini,
        "caixa_final": float(estado.get("caixa_final", 0)),
        "patrimonio_final": patr_fim,
        "lucro_reais": round(lucro, 2),
        "retorno_pct": round(ret_pct, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "stop_loss_pct": estado.get("stop_loss_pct"),
        "take_profit_pct": estado.get("take_profit_pct"),
        "trailing_stop": estado.get("trailing_stop"),
        "risco_trade_pct": estado.get("risco_trade_pct"),
        "universo": estado.get("universo"),
        "total_ordens": len(ordens),
        "ordens": ordens,
        "posicoes_abertas": estado.get("posicoes_abertas", []),
        "historico": estado.get("historico", [])
    }
    
    idx = -1
    for i, item in enumerate(banco):
        if item.get("id_ciclo") == id_ciclo:
            idx = i
            break
    if idx >= 0:
        banco[idx] = registro
    else:
        banco.append(registro)
        
    try:
        with open(ARQUIVO_BANCO_HISTORICO, "w", encoding="utf-8") as f:
            json.dump(banco, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 6. Barra Lateral (Parâmetros Globais)
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Painel de Controle B3")

status_texto, status_sub, hora_sp = obter_status_mercado_b3()
st.sidebar.markdown(f"**Relógio B3 (Brasília):** `{hora_sp.strftime('%H:%M:%S')}`")
st.sidebar.markdown(f"**Status:** {status_texto}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Pesos de Avaliação Quantitativa")
peso_lucro = st.sidebar.slider("Peso do Lucro / Variação (%)", 0, 100, 50, step=5) / 100
peso_liquidez = st.sidebar.slider("Peso da Liquidez Diária (%)", 0, 100, 50, step=5) / 100

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Sincronizar B3 Agora", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Operando sob regras de pregão da Bolsa B3 (10h às 17h, D+2 de liquidação).")

# -----------------------------------------------------------------------------
# 6. Cabeçalho e Painel Macroeconômico
# -----------------------------------------------------------------------------
macro = obter_dados_macro()

st.title("🤖 Plataforma Quantitativa B3: Robô Trader, Carteiras & Sinais")
st.markdown(f"**Sistema Autônomo com Regras Oficiais de Pregão da B3** | Horário de Brasília: `{hora_sp.strftime('%d/%m/%Y %H:%M:%S')}`")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Taxa Selic Meta", f"{macro['selic']:.2f}% a.a.", help="SGS Banco Central")
with col_m2:
    st.metric("CDI Referência", f"{macro['cdi']:.2f}% a.a.", help="Benchmark financeiro")
with col_m3:
    st.metric("Status do Pregão B3", status_texto.split(" ")[1] if " " in status_texto else status_texto, delta=status_sub)
with col_m4:
    st.metric("Horário de Negociação", "10:00 às 17:00 (Regular)")

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. Estrutura de Abas (Com o Robô Trader em Destaque)
# -----------------------------------------------------------------------------
aba_robo, aba_carteiras, aba_simulador, aba_tickers, aba_matriz = st.tabs([
    "🤖 Robô Trader B3 (Simulador Autônomo)",
    "🎯 Carteiras Recomendadas",
    "🧮 Simulador de Investimentos",
    "📋 Todos os Tickers B3 (+2.300 ativos com Sinais)",
    "📊 Matriz de Oportunidades & Quadrantes"
])

# =============================================================================
# ABA 1: ROBÔ TRADER B3 (SIMULADOR AUTÔNOMO COM HORÁRIO E RELATÓRIOS)
# =============================================================================
with aba_robo:
    st.subheader("🤖 Robô Trader B3: Corretor Autônomo com Ciclo Programado")
    st.markdown(
        "Configure o robô, defina a **Data de Início (ex: hoje)** e a **Data de Desligamento (ex: dia 15)**. "
        "Ao ligar, o robô opera de forma autônoma na B3, persiste o estado em disco e disponibiliza o **Relatório Final Consolidado** ao término do ciclo."
    )

    estado_robo = carregar_estado_robo()

    # Se o robô estiver ativo, verificar se já atingiu a data de encerramento
    if estado_robo is not None and estado_robo.get("status") == "ATIVO":
        dt_fim_obj = datetime.strptime(estado_robo["data_fim"], "%Y-%m-%d").date()
        hoje_sp = datetime.now(pytz.timezone("America/Sao_Paulo")).date()
        if hoje_sp >= dt_fim_obj:
            estado_robo["status"] = "ENCERRADO"
            estado_robo["data_encerramento"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            salvar_estado_robo(estado_robo)

    # =========================================================================
    # ESTADO 1: INATIVO (Painel de Configuração para Ligar o Robô)
    # =========================================================================
    if estado_robo is None:
        st.info("⚪ **Status: Robô Desligado**. Configure os parâmetros abaixo e clique em **Ligar Robô Trader** para iniciar o ciclo.")

        with st.expander("⚙️ Parâmetros do Ciclo Operacional", expanded=True):
            c_r1, c_r2, c_r3 = st.columns(3)
            with c_r1:
                capital_robo = st.number_input("Capital Inicial a Alocar (R$):", min_value=1_000.0, max_value=10_000_000.0, value=50_000.0, step=5_000.0)
                estrategia_robo = st.selectbox(
                    "Estratégia Algorítmica Avançada:",
                    [
                        "Confluência Multi-Indicador (Profissional)",
                        "MACD + Bollinger (Volatilidade)",
                        "Momentum com ADX (Força de Tendência)",
                        "IFR + Estocástico (Reversão à Média)",
                        "VWAP + OBV (Fluxo Institucional)",
                        "Cruzamento de Médias (EMA 9 x EMA 21)",
                        "IFR / RSI Sobrevendido"
                    ],
                    index=0
                )
                opcoes_universo = {
                    "Top 10 Blue Chips de Alta Liquidez": ["PETR4", "VALE3", "ITUB4", "BBAS3", "WEGE3", "PRIO3", "RENT3", "GGBR4", "SUZB3", "B3SA3"],
                    "Top 20 Ações Mais Negociadas da B3": ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "ABEV3", "RENT3", "PRIO3", "SUZB3", "GGBR4", "RADL3", "LREN3", "EQTL3", "SBSP3", "TOTS3", "UGPA3", "EGIE3", "CYRE3", "B3SA3"],
                    "Ações de Dividendos e Renda": ["BBAS3", "ITUB4", "BBDC4", "TAEE11", "CPFE3", "CMIG4", "VBBR3", "EGIE3", "SANB11"]
                }
                escolha_universo = st.selectbox("Universo de Ações Monitoradas:", list(opcoes_universo.keys()), index=0)
                tickers_selecionados = opcoes_universo[escolha_universo]

            with c_r2:
                hoje_padrao = datetime.now(pytz.timezone("America/Sao_Paulo")).date()
                fim_padrao = hoje_padrao + timedelta(days=3)  # default: 3 dias no futuro (ex: dia 15 se hoje for 12)
                data_inicio = st.date_input("Data de Início das Operações:", value=hoje_padrao)
                data_fim = st.date_input("Data de Desligamento / Encerramento:", value=fim_padrao)
                trailing_stop = st.checkbox("Ativar Trailing Stop Dinâmico (ATR)", value=True)

            with c_r3:
                stop_loss = st.slider("Stop Loss Máximo (%):", min_value=1.0, max_value=15.0, value=3.5, step=0.5)
                take_profit = st.slider("Take Profit / Alvo (%):", min_value=2.0, max_value=30.0, value=7.0, step=0.5)
                risco_trade = st.slider("Teto por Operação (Kelly %):", min_value=10.0, max_value=40.0, value=20.0, step=5.0)

            st.caption(f"Radar operacional ({len(tickers_selecionados)} ativos): {', '.join(tickers_selecionados)}")

        if st.button("🟢 LIGAR ROBÔ TRADER (Iniciar Ciclo Autônomo)", use_container_width=True, type="primary"):
            with st.spinner("Ligando o Robô Trader, registrando ciclo no sistema e sincronizando primeiros pregões..."):
                res_robo = executar_robo_trader(
                    capital_inicial=capital_robo,
                    dt_inicio=data_inicio,
                    dt_fim=data_fim,
                    estrategia=estrategia_robo,
                    stop_loss_pct=stop_loss,
                    take_profit_pct=take_profit,
                    universo_tickers=tickers_selecionados,
                    trailing_stop_ativo=trailing_stop,
                    risco_por_trade_pct=risco_trade
                )

                if res_robo and not res_robo["historico"].empty:
                    novo_estado = {
                        "status": "ATIVO",
                        "capital_inicial": float(capital_robo),
                        "data_inicio": data_inicio.strftime("%Y-%m-%d"),
                        "data_fim": data_fim.strftime("%Y-%m-%d"),
                        "estrategia": estrategia_robo,
                        "stop_loss_pct": float(stop_loss),
                        "take_profit_pct": float(take_profit),
                        "trailing_stop": bool(trailing_stop),
                        "risco_trade_pct": float(risco_trade),
                        "universo": tickers_selecionados,
                        "data_ligado": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "caixa_final": float(res_robo["caixa_final"]),
                        "patrimonio_final": float(res_robo["patrimonio_final"]),
                        "historico": res_robo["historico"].to_dict(orient="records"),
                        "ordens": res_robo["ordens"].to_dict(orient="records"),
                        "posicoes_abertas": res_robo["posicoes_abertas"].to_dict(orient="records")
                    }
                    # Se hoje já for a data de fim ou posterior, já encerra automaticamente
                    hoje_atual = datetime.now(pytz.timezone("America/Sao_Paulo")).date()
                    if hoje_atual >= data_fim:
                        novo_estado["status"] = "ENCERRADO"
                        novo_estado["data_encerramento"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                    salvar_estado_robo(novo_estado)
                    st.success("✅ Robô Trader LIGADO com sucesso! Estado persistido em disco.")
                    st.rerun()
                else:
                    st.error("Erro ao sincronizar com a B3. Verifique a conexão com a internet e tente novamente.")

    # =========================================================================
    # ESTADO 2: ATIVO (Robô em Operação Contínua)
    # =========================================================================
    elif estado_robo.get("status") == "ATIVO":
        st.success(
            f"🟢 **STATUS: ROBÔ TRADER EM OPERAÇÃO ATIVA NA B3** | "
            f"Ciclo Programado: de **{estado_robo['data_inicio']}** até **{estado_robo['data_fim']}** | "
            f"Estratégia: **{estado_robo['estrategia']}**"
        )

        col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
        with col_act1:
            if st.button("🔄 Sincronizar Agora com a B3", use_container_width=True):
                with st.spinner("Buscando cotações atualizadas e recalculando posições..."):
                    dt_ini = datetime.strptime(estado_robo["data_inicio"], "%Y-%m-%d").date()
                    dt_fim = datetime.strptime(estado_robo["data_fim"], "%Y-%m-%d").date()
                    res_sync = executar_robo_trader(
                        capital_inicial=estado_robo["capital_inicial"],
                        dt_inicio=dt_ini,
                        dt_fim=dt_fim,
                        estrategia=estado_robo["estrategia"],
                        stop_loss_pct=estado_robo["stop_loss_pct"],
                        take_profit_pct=estado_robo["take_profit_pct"],
                        universo_tickers=estado_robo["universo"],
                        trailing_stop_ativo=estado_robo["trailing_stop"],
                        risco_por_trade_pct=estado_robo["risco_trade_pct"]
                    )
                    if res_sync:
                        estado_robo["caixa_final"] = float(res_sync["caixa_final"])
                        estado_robo["patrimonio_final"] = float(res_sync["patrimonio_final"])
                        estado_robo["historico"] = res_sync["historico"].to_dict(orient="records")
                        estado_robo["ordens"] = res_sync["ordens"].to_dict(orient="records")
                        estado_robo["posicoes_abertas"] = res_sync["posicoes_abertas"].to_dict(orient="records")
                        salvar_estado_robo(estado_robo)
                        st.rerun()

        with col_act2:
            if st.button("⏹️ Encerrar Ciclo Antecipadamente", use_container_width=True):
                estado_robo["status"] = "ENCERRADO"
                estado_robo["data_encerramento"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S (Antecipado)")
                salvar_estado_robo(estado_robo)
                st.rerun()

        with col_act3:
            if st.button("⚠️ Resetar / Cancelar Robô", use_container_width=True):
                resetar_ciclo_robo()
                st.rerun()

        st.caption(f"💡 O robô está ligado e persistido no arquivo local `robo_state.json`. Na data de **{estado_robo['data_fim']}**, o ciclo será concluído automaticamente e o Relatório Final estará consolidado.")

        # Painel em Tempo Real
        df_hist = pd.DataFrame(estado_robo.get("historico", []))
        df_ordens = pd.DataFrame(estado_robo.get("ordens", []))
        df_pos = pd.DataFrame(estado_robo.get("posicoes_abertas", []))
        cap_ini = estado_robo["capital_inicial"]
        patr_atual = estado_robo.get("patrimonio_final", cap_ini)
        lucro_atual = patr_atual - cap_ini
        ret_atual = (lucro_atual / cap_ini) * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Patrimônio Atual", f"R$ {patr_atual:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), delta=f"{ret_atual:+.2f}%")
        with k2:
            st.metric("Lucro / Prejuízo em Aberto", f"R$ {lucro_atual:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        with k3:
            st.metric("Caixa Disponível", f"R$ {estado_robo.get('caixa_final', 0):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        with k4:
            st.metric("Total de Ordens Executadas", f"{len(df_ordens)} ordens B3")

        if not df_hist.empty:
            fig_eq = px.line(df_hist, x="Data", y="Patrimônio Total (R$)", title="Evolução do Patrimônio em Tempo Real (R$)", markers=True)
            fig_eq.add_hline(y=cap_ini, line_dash="dash", line_color="gray", annotation_text="Capital Inicial")
            fig_eq.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)

        c_p1, c_p2 = st.columns([1, 1])
        with c_p1:
            st.markdown("#### 💼 Posições Abertas em Carteira")
            if not df_pos.empty:
                st.dataframe(df_pos.style.format({"Preço Compra (R$)": "R$ {:.2f}", "Preço Atual (R$)": "R$ {:.2f}", "Valor em Custódia (R$)": "R$ {:.2f}", "Resultado Aberto (R$)": "{:+.2f}", "Retorno (%)": "{:+.2f}%"}), hide_index=True, use_container_width=True)
            else:
                st.info("Nenhuma posição aberta no momento. O robô está 100% em caixa aguardando novo sinal de compra.")

        with c_p2:
            st.markdown("#### 📜 Livro de Ordens em Tempo Real")
            if not df_ordens.empty:
                st.dataframe(df_ordens[["Data Pregão", "Ticker", "Operação", "Qtd", "Preço Execução (R$)", "Resultado Trade (R$)", "Motivo Execução"]].tail(10).style.format({"Preço Execução (R$)": "R$ {:.2f}", "Resultado Trade (R$)": "{:+.2f}"}), hide_index=True, use_container_width=True)
            else:
                st.info("Aguardando abertura de novas posições no horário do pregão.")

    # =========================================================================
    # ESTADO 3: ENCERRADO (Relatório Final Consolidado do Período)
    # =========================================================================
    elif estado_robo.get("status") == "ENCERRADO":
        st.warning(
            f"🏁 **STATUS: CICLO OPERACIONAL CONCLUÍDO** | "
            f"Período operado: de **{estado_robo['data_inicio']}** até **{estado_robo['data_fim']}** | "
            f"Data de Encerramento: **{estado_robo.get('data_encerramento', 'Encerramento Automático')}**"
        )

        df_hist = pd.DataFrame(estado_robo.get("historico", []))
        df_ordens = pd.DataFrame(estado_robo.get("ordens", []))
        cap_ini = estado_robo["capital_inicial"]
        patr_fim = estado_robo.get("patrimonio_final", cap_ini)
        lucro_fim = patr_fim - cap_ini
        ret_fim = (lucro_fim / cap_ini) * 100

        vendas = df_ordens[df_ordens["Operação"] == "VENDA"] if not df_ordens.empty else pd.DataFrame()
        total_trades = len(vendas)
        trades_venc = len(vendas[vendas["Resultado Trade (R$)"] > 0]) if not vendas.empty else 0
        win_rate = (trades_venc / total_trades * 100) if total_trades > 0 else 0.0

        lucro_ganhos = vendas[vendas["Resultado Trade (R$)"] > 0]["Resultado Trade (R$)"].sum() if not vendas.empty else 0
        prej_perdas = abs(vendas[vendas["Resultado Trade (R$)"] < 0]["Resultado Trade (R$)"].sum()) if not vendas.empty else 0
        profit_factor = (lucro_ganhos / prej_perdas) if prej_perdas > 0 else (lucro_ganhos if lucro_ganhos > 0 else 1.0)

        if not df_hist.empty:
            df_hist["Pico"] = df_hist["Patrimônio Total (R$)"].cummax()
            df_hist["Drawdown"] = (df_hist["Patrimônio Total (R$)"] - df_hist["Pico"]) / df_hist["Pico"] * 100
            max_dd = df_hist["Drawdown"].min()
        else:
            max_dd = 0.0

        st.subheader("📑 Relatório Final Consolidado do Corretor B3")
        
        st.success(
            f"**Parecer do Corretor Especialista B3**: O ciclo foi finalizado com resultado líquido de "
            f"**R$ {lucro_fim:,.2f} ({ret_fim:+.2f}%)**, totalizando **{total_trades} trades fechados**, "
            f"taxa de acerto de **{win_rate:.1f}%** e Fator de Lucro de **{profit_factor:.2f}**. Todos os emolumentos da B3 (0,03% por ordem) foram computados."
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Patrimônio Final", f"R$ {patr_fim:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), delta=f"{ret_fim:+.2f}%")
        with k2:
            st.metric("Lucro Líquido Real", f"R$ {lucro_fim:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        with k3:
            st.metric("Taxa de Acerto (Win Rate)", f"{win_rate:.1f}%")
        with k4:
            st.metric("Fator de Lucro (Profit Factor)", f"{profit_factor:.2f}")
        with k5:
            st.metric("Drawdown Máximo", f"{max_dd:.2f}%")

        if not df_hist.empty:
            fig_eq = px.line(df_hist, x="Data", y="Patrimônio Total (R$)", title=f"Curva de Capital Final do Ciclo ({estado_robo['data_inicio']} a {estado_robo['data_fim']})", markers=True)
            fig_eq.add_hline(y=cap_ini, line_dash="dash", line_color="gray", annotation_text="Capital Inicial")
            fig_eq.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_eq, use_container_width=True)

        st.markdown("#### 📜 Extrato Oficial de Todas as Ordens Executadas no Ciclo")
        if not df_ordens.empty:
            st.dataframe(df_ordens.style.format({"Preço Execução (R$)": "R$ {:.2f}", "Taxas B3 (R$)": "R$ {:.2f}", "Resultado Trade (R$)": "{:+.2f}", "Retorno (%)": "{:+.2f}%"}), hide_index=True, use_container_width=True)
            csv_relatorio = df_ordens.to_csv(index=False).encode('utf-8')
            c_d1, c_d2 = st.columns([1, 1])
            with c_d1:
                st.download_button(
                    label="📥 Baixar Relatório Final Oficial em CSV",
                    data=csv_relatorio,
                    file_name=f"relatorio_final_robo_b3_{estado_robo['data_inicio']}_a_{estado_robo['data_fim']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c_d2:
                if st.button("🚀 Iniciar Novo Ciclo Operacional", use_container_width=True, type="primary"):
                    resetar_ciclo_robo()
                    st.rerun()

    # =========================================================================
    # BANCO HISTÓRICO PERMANENTE DE CICLOS DO ROBÔ (Para Comparações Futuras)
    # =========================================================================
    banco_historico = carregar_banco_historico()
    if banco_historico:
        st.markdown("---")
        st.subheader("📚 Banco Histórico de Ciclos & Comparativo de Estratégias")
        st.markdown(
            "Todos os ciclos operacionais executados pelo robô ficam registrados abaixo para **auditoria, análise de performance e comparações futuras** entre estratégias."
        )

        dados_tabela_banco = []
        for reg in banco_historico:
            dados_tabela_banco.append({
                "ID Ciclo": reg.get("id_ciclo"),
                "Data Registro": reg.get("data_registro"),
                "Estratégia": reg.get("estrategia"),
                "Período Operado": f"{reg.get('data_inicio')} a {reg.get('data_fim')}",
                "Status": reg.get("status"),
                "Capital Inicial (R$)": reg.get("capital_inicial", 0.0),
                "Patrimônio Final (R$)": reg.get("patrimonio_final", 0.0),
                "Lucro Líquido (R$)": reg.get("lucro_reais", 0.0),
                "Retorno (%)": reg.get("retorno_pct", 0.0),
                "Win Rate (%)": reg.get("win_rate_pct", 0.0),
                "Profit Factor": reg.get("profit_factor", 0.0),
                "Total Ordens": reg.get("total_ordens", 0)
            })

        df_banco = pd.DataFrame(dados_tabela_banco)
        st.dataframe(
            df_banco.style.format({
                "Capital Inicial (R$)": "R$ {:,.2f}",
                "Patrimônio Final (R$)": "R$ {:,.2f}",
                "Lucro Líquido (R$)": "R$ {:+,.2f}",
                "Retorno (%)": "{:+.2f}%",
                "Win Rate (%)": "{:.1f}%",
                "Profit Factor": "{:.2f}"
            }),
            hide_index=True,
            use_container_width=True
        )

        # Gráfico comparativo se houver pelo menos 1 ciclo
        if len(df_banco) > 1:
            fig_comp = px.bar(
                df_banco,
                x="Estratégia",
                y="Retorno (%)",
                color="Retorno (%)",
                color_continuous_scale=["red", "yellow", "green"],
                title="Comparativo de Retorno (%) entre Estratégias e Ciclos Testados",
                text="Retorno (%)"
            )
            fig_comp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_comp.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_comp, use_container_width=True)

        # Inspecionar detalhes de um ciclo passado
        with st.expander("🔍 Inspecionar Livro de Ordens de um Ciclo Específico"):
            opcoes_ciclos = [c.get("id_ciclo") for c in banco_historico]
            ciclo_escolhido = st.selectbox("Selecione o Ciclo para Visualizar:", opcoes_ciclos)
            reg_escolhido = next((c for c in banco_historico if c.get("id_ciclo") == ciclo_escolhido), None)
            if reg_escolhido:
                st.write(f"**Estratégia**: {reg_escolhido.get('estrategia')} | **Período**: {reg_escolhido.get('data_inicio')} a {reg_escolhido.get('data_fim')} | **Retorno**: {reg_escolhido.get('retorno_pct'):+.2f}%")
                ordens_ciclo = pd.DataFrame(reg_escolhido.get("ordens", []))
                if not ordens_ciclo.empty:
                    st.dataframe(ordens_ciclo, hide_index=True, use_container_width=True)
                else:
                    st.info("Nenhuma ordem executada neste ciclo.")

        # Download do Banco Histórico Completo
        csv_banco = df_banco.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Banco Histórico Consolidado de Todos os Ciclos (CSV)",
            data=csv_banco,
            file_name=f"historico_geral_ciclos_robo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# =============================================================================
# DEMAIS ABAS DO SISTEMA
# =============================================================================
with aba_carteiras:
    st.subheader("🏆 Carteiras Recomendadas por Perfil de Investidor")
    perfil_sel = st.radio(
        "Selecione a Estratégia:",
        ["🛡️ Conservadora (Máxima Liquidez)", "💰 Dividendos & Renda Passiva", "🚀 Crescimento & Alpha"],
        horizontal=True
    )
    if "Conservadora" in perfil_sel:
        cart_data = [
            {"Ativo": "Tesouro Selic", "Tipo": "Renda Fixa Pública", "Alocação (%)": 40, "Liquidez": "D+1", "Rentabilidade Alvo": f"{macro['selic']:.2f}% a.a.", "Sinal": "🟢 COMPRA FORTE"},
            {"Ativo": "CDB 105% CDI", "Tipo": "Renda Fixa Bancária", "Alocação (%)": 30, "Liquidez": "D+0", "Rentabilidade Alvo": f"~{macro['cdi']*1.05:.2f}% a.a.", "Sinal": "🟢 COMPRA"},
            {"Ativo": "KNCR11", "Tipo": "FII de Papel (CDI)", "Alocação (%)": 15, "Liquidez": "D+2", "Rentabilidade Alvo": "CDI + 2.0% a.a. isento", "Sinal": "🟢 COMPRA"},
            {"Ativo": "HGLG11", "Tipo": "FII de Logística", "Alocação (%)": 15, "Liquidez": "D+2", "Rentabilidade Alvo": "IPCA + 7.5% a.a.", "Sinal": "🟡 MANTER"}
        ]
    elif "Dividendos" in perfil_sel:
        cart_data = [
            {"Ativo": "BBAS3", "Tipo": "Ação (Bancário)", "Alocação (%)": 20, "Liquidez": "Alta", "Rentabilidade Alvo": "DY ~9.5% a.a.", "Sinal": "🟢 COMPRA"},
            {"Ativo": "ITUB4", "Tipo": "Ação (Bancário)", "Alocação (%)": 20, "Liquidez": "Altíssima", "Rentabilidade Alvo": "DY ~7.5% a.a.", "Sinal": "🟢 COMPRA FORTE"},
            {"Ativo": "TAEE11", "Tipo": "Ação (Elétrica)", "Alocação (%)": 15, "Liquidez": "Alta", "Rentabilidade Alvo": "DY ~8.5% a.a.", "Sinal": "🟡 MANTER"},
            {"Ativo": "MXRF11", "Tipo": "FII de Papel", "Alocação (%)": 15, "Liquidez": "Máxima FII", "Rentabilidade Alvo": "DY ~11.0% a.a. isento", "Sinal": "🟢 COMPRA"},
            {"Ativo": "XPML11", "Tipo": "FII Shoppings", "Alocação (%)": 15, "Liquidez": "Alta", "Rentabilidade Alvo": "DY ~8.5% a.a. isento", "Sinal": "🟢 COMPRA"},
            {"Ativo": "Tesouro Selic", "Tipo": "Caixa Oportunidade", "Alocação (%)": 15, "Liquidez": "D+1", "Rentabilidade Alvo": f"{macro['selic']:.2f}% a.a.", "Sinal": "🟢 COMPRA"}
        ]
    else:
        cart_data = [
            {"Ativo": "PETR4", "Tipo": "Ação (Petróleo)", "Alocação (%)": 20, "Liquidez": "Máxima", "Rentabilidade Alvo": "DY ~14% + Caixa", "Sinal": "🟢 COMPRA FORTE"},
            {"Ativo": "VALE3", "Tipo": "Ação (Mineração)", "Alocação (%)": 15, "Liquidez": "Máxima", "Rentabilidade Alvo": "Ciclo Global", "Sinal": "🟢 COMPRA"},
            {"Ativo": "WEGE3", "Tipo": "Ação (Indústria/Tech)", "Alocação (%)": 20, "Liquidez": "Alta", "Rentabilidade Alvo": "CAGR > 15% a.a.", "Sinal": "🟢 COMPRA"},
            {"Ativo": "PRIO3", "Tipo": "Ação (Petróleo Júnior)", "Alocação (%)": 15, "Liquidez": "Alta", "Rentabilidade Alvo": "Crescimento Produção", "Sinal": "🟢 COMPRA FORTE"},
            {"Ativo": "RENT3", "Tipo": "Ação (Mobilidade)", "Alocação (%)": 15, "Liquidez": "Alta", "Rentabilidade Alvo": "Recuperação Ciclo", "Sinal": "🟡 MANTER"},
            {"Ativo": "Tesouro Selic", "Tipo": "Caixa Tático", "Alocação (%)": 15, "Liquidez": "D+1", "Rentabilidade Alvo": f"{macro['selic']:.2f}% a.a.", "Sinal": "🟢 COMPRA"}
        ]
    df_c = pd.DataFrame(cart_data)
    c_p1, c_p2 = st.columns([1, 2])
    with c_p1:
        fig_p = px.pie(df_c, values="Alocação (%)", names="Ativo", hole=0.45, title="Alocação da Carteira")
        st.plotly_chart(fig_p, use_container_width=True)
    with c_p2:
        st.dataframe(df_c, hide_index=True, use_container_width=True)

with aba_simulador:
    st.subheader("🧮 Simulador de Investimentos & Projeção Patrimonial")
    st.caption("Simule a evolução do seu patrimônio comparando a carteira recomendada com o CDI e a Poupança.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        aporte_inicial = st.number_input("Capital Inicial (R$):", min_value=100.0, max_value=10_000_000.0, value=20_000.0, step=1000.0)
    with col_s2:
        aporte_mensal = st.number_input("Aporte Mensal (R$):", min_value=0.0, max_value=1_000_000.0, value=1_500.0, step=200.0)
    with col_s3:
        prazo_meses = st.slider("Prazo de Investimento (Meses):", min_value=6, max_value=120, value=36, step=6)
    with col_s4:
        estrategia_simulacao = st.selectbox(
            "Estratégia Escolhida:",
            ["Crescimento & Alpha (CDI + 6%)", "Dividendos (CDI + 3%)", "Conservadora (CDI + 0.5%)"]
        )

    # Taxas anuais
    if "Alpha" in estrategia_simulacao:
        taxa_carteira_aa = (macro['cdi'] + 6.0) / 100
    elif "Dividendos" in estrategia_simulacao:
        taxa_carteira_aa = (macro['cdi'] + 3.0) / 100
    else:
        taxa_carteira_aa = (macro['cdi'] + 0.5) / 100

    taxa_cdi_aa = macro['cdi'] / 100
    taxa_poupanca_aa = 0.0617  # Poupança clássica ~6.17% a.a.

    # Taxas mensais equivalentes
    r_cart_mes = (1 + taxa_carteira_aa) ** (1/12) - 1
    r_cdi_mes = (1 + taxa_cdi_aa) ** (1/12) - 1
    r_poup_mes = (1 + taxa_poupanca_aa) ** (1/12) - 1

    meses = list(range(prazo_meses + 1))
    val_cart = [aporte_inicial]
    val_cdi = [aporte_inicial]
    val_poup = [aporte_inicial]
    total_investido = [aporte_inicial]

    for m in range(1, prazo_meses + 1):
        total_investido.append(total_investido[-1] + aporte_mensal)
        val_cart.append((val_cart[-1] + aporte_mensal) * (1 + r_cart_mes))
        val_cdi.append((val_cdi[-1] + aporte_mensal) * (1 + r_cdi_mes))
        val_poup.append((val_poup[-1] + aporte_mensal) * (1 + r_poup_mes))

    df_sim = pd.DataFrame({
        "Mês": meses,
        "Total Aportado": total_investido,
        "Carteira Recomendada": val_cart,
        "100% do CDI": val_cdi,
        "Poupança": val_poup
    })

    saldo_final_cart = val_cart[-1]
    saldo_final_cdi = val_cdi[-1]
    total_aportado_final = total_investido[-1]
    lucro_liquido_cart = saldo_final_cart - total_aportado_final
    renda_mensal_estimada = saldo_final_cart * r_cart_mes

    st.markdown("---")
    res1, res2, res3, res4 = st.columns(4)
    with res1:
        st.metric("Total Investido do Bolso", f"R$ {total_aportado_final:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    with res2:
        st.metric("Patrimônio Final na Carteira", f"R$ {saldo_final_cart:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), delta=f"+ R$ {saldo_final_cart - saldo_final_cdi:,.2f} vs. CDI".replace(",", "X").replace(".", ",").replace("X", "."))
    with res3:
        st.metric("Lucro Líquido dos Rendimentos", f"R$ {lucro_liquido_cart:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    with res4:
        st.metric("Renda Passiva Mensal Estimada", f"R$ {renda_mensal_estimada:,.2f}/mês".replace(",", "X").replace(".", ",").replace("X", "."), help="Rendimento mensal ao término do período")

    # Gráfico de evolução patrimonial
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=df_sim["Mês"], y=df_sim["Carteira Recomendada"], mode="lines", name=f"Carteira ({estrategia_simulacao.split(' ')[0]})", line=dict(color="#00E676", width=3)))
    fig_sim.add_trace(go.Scatter(x=df_sim["Mês"], y=df_sim["100% do CDI"], mode="lines", name=f"100% do CDI ({macro['cdi']:.2f}% a.a.)", line=dict(color="#29B6F6", width=2, dash="dash")))
    fig_sim.add_trace(go.Scatter(x=df_sim["Mês"], y=df_sim["Poupança"], mode="lines", name="Poupança (~6.17% a.a.)", line=dict(color="#FFA726", width=2, dash="dot")))
    fig_sim.add_trace(go.Scatter(x=df_sim["Mês"], y=df_sim["Total Aportado"], mode="lines", name="Total Aportado do Bolso", line=dict(color="#9E9E9E", width=1)))

    fig_sim.update_layout(
        title="Projeção Comparativa de Evolução Patrimonial (R$)",
        xaxis_title="Tempo (Meses)",
        yaxis_title="Patrimônio Acumulado (R$)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# Coleta de Tickers para Aba 4 e 5
with st.spinner("Atualizando lista de ativos da B3..."):
    df_raw = carregar_todos_tickers_b3()

with aba_tickers:
    st.subheader("📋 Grade de Tickers da B3 com Sinais de Compra/Venda")
    if not df_raw.empty:
        df_t = df_raw.copy()
        
        # Sinais rápidos
        def gerar_sinal(row):
            var = row["Variação Dia (%)"]
            vol = row["Volume Financeiro (R$ Mi)"]
            if var > 1.0 and vol > 1.0:
                return "🟢 COMPRA FORTE"
            elif var >= 0.0 and vol > 0.1:
                return "🟢 COMPRA"
            elif var > -1.5:
                return "🟡 MANTER"
            elif var <= -2.5 and vol > 1.0:
                return "🔴 VENDA / STOP"
            else:
                return "⚪ AGUARDAR / EVITAR"

        df_t["Sinal Operacional"] = df_t.apply(gerar_sinal, axis=1)

        col_pesq1, col_pesq2 = st.columns([2, 1])
        with col_pesq1:
            busca = st.text_input("🔍 Pesquisar Ativo (Código ou Nome):", "").strip().upper()
        with col_pesq2:
            filtro_s = st.selectbox("Filtrar por Sinal:", ["Todos os Sinais", "🟢 COMPRA FORTE", "🟢 COMPRA", "🟡 MANTER", "🔴 VENDA / STOP", "⚪ AGUARDAR / EVITAR"])

        if busca:
            df_t = df_t[df_t["Ticker"].str.contains(busca) | df_t["Nome / Razão Social"].str.upper().str.contains(busca)]
        if filtro_s != "Todos os Sinais":
            df_t = df_t[df_t["Sinal Operacional"] == filtro_s]

        colunas_grid = ["Ticker", "Nome / Razão Social", "Classe", "Preço Atual (R$)", "Variação Dia (%)", "Volume Financeiro (R$ Mi)", "Sinal Operacional"]
        st.dataframe(
            df_t[colunas_grid].style.format({
                "Preço Atual (R$)": "R$ {:.2f}",
                "Variação Dia (%)": "{:+.2f}%",
                "Volume Financeiro (R$ Mi)": "R$ {:.2f} Mi"
            }),
            use_container_width=True,
            hide_index=True,
            height=500
        )

        csv_t = df_t[colunas_grid].to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Baixar Grade de Tickers ({len(df_t)} ativos) em CSV",
            data=csv_t,
            file_name=f"tickers_b3_sinais_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with aba_matriz:
    st.subheader("📊 Matriz de Oportunidades: Lucratividade x Liquidez")
    if not df_raw.empty:
        df_top = df_raw.sort_values(by="Volume Financeiro (R$ Mi)", ascending=False).head(60)
        fig_scat = px.scatter(
            df_top,
            x="Volume Financeiro (R$ Mi)",
            y="Variação Dia (%)",
            text="Ticker",
            color="Classe",
            hover_name="Nome / Razão Social",
            size="Volume Financeiro (R$ Mi)",
            labels={"Volume Financeiro (R$ Mi)": "Volume Negociado (R$ Milhões)", "Variação Dia (%)": "Lucratividade / Variação Dia (%)"}
        )
        fig_scat.update_traces(textposition='top center')
        fig_scat.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scat, use_container_width=True)

