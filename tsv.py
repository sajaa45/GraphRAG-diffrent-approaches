import pandas as pd

#df = pd.read_csv(r"C:\Users\LENOVO\Downloads\sub.txt", sep="\t")
#c=df["adsh"].nunique()
#counts = df["adsh"].value_counts()
##metrices = df["tag"].value_counts()
##m=df["tag"].nunique()
#type = df["sic"].value_counts()
#t=df["sic"].nunique()
#
#print(counts)
#print(type)
#print("companies:",c)
#print("type:",t)
# Method 2: Case-insensitive search (recommended)
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\num.txt", sep="\t")
risk_tags = df[df["tag"].str.contains("risk", case=False, na=False)]
risk_counts = risk_tags["tag"].value_counts()
print("Tags containing 'risk' and their frequencies:")
print(risk_counts[:10])

# Optional: Print total number of occurrences
print(f"\nTotal risk-related tags: {risk_counts.nunique()}")