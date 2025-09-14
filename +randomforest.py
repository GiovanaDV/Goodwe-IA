import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1) Carregar dados
# ===============================
caminho = "C:\\Users\\giova\\Downloads\\Plant Power_20250913042717.xls"

# Tenta abrir pulando 2 linhas (os arquivos da GoodWe têm metadados no início)
df = pd.read_excel(caminho, header=2, engine="xlrd")

# Padronizar nomes das colunas: tudo minúsculo, sem espaços extras
df.columns = df.columns.str.strip().str.lower()

# Mostrar as colunas para conferência
print("\n📌 Colunas disponíveis no arquivo:")
print(df.columns.tolist())

# ===============================
# 2) Preparar os dados
# ===============================
# Converter coluna de tempo
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Renomear a coluna de consumo para um nome padrão
for col in df.columns:
    if "load" in col:   # procura qualquer coluna que tenha "load"
        df.rename(columns={col: "load"}, inplace=True)

# Remover valores nulos
df = df.dropna(subset=["time", "load"])

# Garantir que load é numérico
df["load"] = pd.to_numeric(df["load"], errors="coerce")

# ===============================
# 3) Selecionar 1 dia específico
# ===============================
dia = "12.09.2025"
df_dia = df[df["time"].dt.date == pd.to_datetime(dia).date()]

if df_dia.empty:
    raise ValueError(f"Nenhum dado encontrado para o dia {dia}")

# ===============================
# 4) Consumo real
# ===============================
plt.figure(figsize=(12, 5))
plt.plot(df_dia["time"], df_dia["load"], label="Consumo real")
plt.title(f"Consumo real no dia {dia}")
plt.ylabel("Consumo (W)")
plt.xlabel("Hora")
plt.legend()
plt.show()

# ===============================
# 5) Estimar standby (média da madrugada 0h-5h)
# ===============================
standby = df_dia[df_dia["time"].dt.hour.between(0, 5)]["load"].mean()

# Criar cenário otimizado
df_dia["load_optimized"] = (df_dia["load"] - standby).clip(lower=0)

# ===============================
# 6) Comparar cenários (gráfico)
# ===============================
plt.figure(figsize=(12, 5))
plt.plot(df_dia["time"], df_dia["load"], label="Cenário real", color="red")
plt.plot(df_dia["time"], df_dia["load_optimized"], label="Cenário otimizado", color="green")
plt.title(f"Comparação de consumo no dia {dia}")
plt.ylabel("Consumo (W)")
plt.xlabel("Hora")
plt.legend()
plt.show()

# ===============================
# 7) Calcular economia
# ===============================
intervalo_horas = 5 / 60  # cada linha é 5 minutos
energia_real_kwh = (df_dia["load"].sum() * intervalo_horas) / 1000
energia_otimizada_kwh = (df_dia["load_optimized"].sum() * intervalo_horas) / 1000
economia_kwh = energia_real_kwh - energia_otimizada_kwh


print("\n📊 RESULTADOS")
print("Consumo real:", round(energia_real_kwh, 2), "kWh")
print("Consumo otimizado:", round(energia_otimizada_kwh, 2), "kWh")


# implementando randomforest - machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# 8) Preparar dados para ML
# ===============================
# Criar feature de hora
df["hora"] = df["time"].dt.hour

# Definir X (features) e y (target)
features = ["hora"]
for col in df.columns:
    if "pv" in col or "battery" in col or "grid" in col:
        features.append(col)

X = df[features]
y = df["load"]

# Como só temos 1 dia, separamos treino (70%) e teste (30%) manualmente
split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ===============================
# 9) Treinar RandomForest
# ===============================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Previsão
y_pred = rf.predict(X_test)

print("\n📈 Avaliação do modelo RandomForest")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R²:", round(r2_score(y_test, y_pred), 2))

# ===============================
# 10) Detectar períodos de standby
# ===============================
df_dia = df.copy()  # já é só um dia
df_dia["previsto"] = rf.predict(X)

# Se consumo real for MAIOR que previsto + tolerância, consideramos standby ex

# ===============================
# 11) Mostrar quando a IA atuou
# ===============================
# Previsão do consumo
df_dia["previsto"] = rf.predict(X)

# Criar coluna standby_detectado
tolerancia = 50  # watts
df_dia["standby_detectado"] = df_dia["load"] > (df_dia["previsto"] + tolerancia)

# Simular desligamento do standby
df_dia["load_ai"] = df_dia.apply(
    lambda row: row["previsto"] if row["standby_detectado"] else row["load"], axis=1
)

if "standby_detectado" not in df_dia.columns:
    raise ValueError("A coluna 'standby_detectado' não foi criada. Verifique o passo anterior.")

print("\n⚡ Ações da IA (detecção de standby):")
for idx, row in df_dia.iterrows():
    hora = row["time"].strftime("%H:%M")
    status = "IA DESLIGOU standby" if row["standby_detectado"] else "Nada a fazer"
    print(f"{hora} → {status}")


plt.figure(figsize=(12, 5))
plt.plot(df_dia["time"], df_dia["load"], label="Consumo real", color="red")
plt.plot(df_dia["time"], df_dia["load_ai"], label="Consumo otimizado pela IA", color="green")

# Marcar pontos onde a IA desligou standby
acoes = df_dia[df_dia["standby_detectado"]]
plt.scatter(acoes["time"], acoes["load"], color="blue", marker="o", label="IA desligou standby")

plt.title("Consumo real vs otimizado com atuação da IA")
plt.ylabel("Consumo (W)")
plt.xlabel("Hora")
plt.legend()
plt.show()
