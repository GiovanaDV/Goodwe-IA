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

