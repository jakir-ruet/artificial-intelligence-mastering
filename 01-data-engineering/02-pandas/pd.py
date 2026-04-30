import pandas as pd

s = pd.Series([25, 35, 62, 57, 50])
print(s)

data = {
	"Name": ["Jakir", "Samim"],
	"Age": [25, 35]
}

df = pd.DataFrame(data)
print(df)
