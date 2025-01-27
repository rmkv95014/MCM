import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer


noc_to_country = {
  "AFG": "Afghanistan",
  "ALB": "Albania",
  "ALG": "Algeria",
  "AND": "Andorra",
  "ANG": "Angola",
  "ANT": "Antigua and Barbuda",
  "ARG": "Argentina",
  "ARM": "Armenia",
  "ARU": "Aruba",
  "ASA": "American Samoa",
  "AUS": "Australia",
  "AUT": "Austria",
  "AZE": "Azerbaijan",
  "BAH": "Bahamas",
  "BAN": "Bangladesh",
  "BAR": "Barbados",
  "BDI": "Burundi",
  "BEL": "Belgium",
  "BEN": "Benin",
  "BER": "Bermuda",
  "BHU": "Bhutan",
  "BIH": "Bosnia and Herzegovina",
  "BIZ": "Belize",
  "BLR": "Belarus",
  "BOL": "Bolivia",
  "BOT": "Botswana",
  "BRA": "Brazil",
  "BRN": "Bahrain",
  "BRU": "Brunei",
  "BUL": "Bulgaria",
  "BUR": "Burkina Faso",
  "CAF": "Central African Republic",
  "CAM": "Cambodia",
  "CAN": "Canada",
  "CAY": "Cayman Islands",
  "CGO": "Congo",
  "CHA": "Chad",
  "CHI": "Chile",
  "CHN": "China",
  "CIV": "Ivory Coast",
  "CMR": "Cameroon",
  "COD": "Democratic Republic of the Congo",
  "COK": "Cook Islands",
  "COL": "Colombia",
  "COM": "Comoros",
  "CPV": "Cape Verde",
  "CRC": "Costa Rica",
  "CRO": "Croatia",
  "CUB": "Cuba",
  "CYP": "Cyprus",
  "CZE": "Czech Republic",
  "DEN": "Denmark",
  "DJI": "Djibouti",
  "DMA": "Dominica",
  "DOM": "Dominican Republic",
  "ECU": "Ecuador",
  "UAR": "Egypt",
  "EGY": "Egypt",
  "ERI": "Eritrea",
  "ESA": "El Salvador",
  "ESP": "Spain",
  "EST": "Estonia",
  "ETH": "Ethiopia",
  "FIJ": "Fiji",
  "FIN": "Finland",
  "FRA": "France",
  "FSM": "Micronesia",
  "GAB": "Gabon",
  "GAM": "Gambia",
  "GBR": "Great Britain",
  "GBS": "Guinea-Bissau",
  "GEO": "Georgia",
  "GEQ": "Equatorial Guinea",
  "GDR": "Germany",
  "SAA": "Germany",
  "FRG": "Germany",
  "GER": "Germany",
  "GHA": "Ghana",
  "GRE": "Greece",
  "GRN": "Grenada",
  "GUA": "Guatemala",
  "GUI": "Guinea",
  "GUM": "Guam",
  "GUY": "Guyana",
  "HAI": "Haiti",
  "HKG": "Hong Kong",
  "HON": "Honduras",
  "HUN": "Hungary",
  "INA": "Indonesia",
  "IND": "India",
  "IRI": "Iran",
  "IRL": "Ireland",
  "IRQ": "Iraq",
  "ISL": "Iceland",
  "ISR": "Israel",
  "ISV": "Virgin Islands",
  "ITA": "Italy",
  "IVB": "British Virgin Islands",
  "JAM": "Jamaica",
  "JOR": "Jordan",
  "JPN": "Japan",
  "KAZ": "Kazakhstan",
  "KEN": "Kenya",
  "KGZ": "Kyrgyzstan",
  "KIR": "Kiribati",
  "KOR": "South Korea",
  "KOS": "Kosovo",
  "KSA": "Saudi Arabia",
  "KUW": "Kuwait",
  "LAO": "Laos",
  "LAT": "Latvia",
  "LBA": "Libya",
  "LBR": "Liberia",
  "LCA": "Saint Lucia",
  "LES": "Lesotho",
  "LEB": "Lebanon",
  "LBN": "Lebanon",
  "LIB": "Lebanon",
  "LIE": "Liechtenstein",
  "LTU": "Lithuania",
  "LUX": "Luxembourg",
  "MAD": "Madagascar",
  "MAR": "Morocco",
  "MAL": "Malaysia",
  "MAS": "Malaysia",
  "MAW": "Malawi",
  "MDA": "Moldova",
  "MDV": "Maldives",
  "MEX": "Mexico",
  "MGL": "Mongolia",
  "MHL": "Marshall Islands",
  "MKD": "North Macedonia",
  "MLI": "Mali",
  "MLT": "Malta",
  "MNE": "Montenegro",
  "MON": "Monaco",
  "MOZ": "Mozambique",
  "MRI": "Mauritius",
  "MTN": "Mauritania",
  "MYA": "Myanmar",
  "NAM": "Namibia",
  "NCA": "Nicaragua",
  "NED": "Netherlands",
  "NEP": "Nepal",
  "NGR": "Nigeria",
  "NIG": "Niger",
  "NOR": "Norway",
  "NRU": "Nauru",
  "NZL": "New Zealand",
  "OMA": "Oman",
  "PAK": "Pakistan",
  "PAN": "Panama",
  "PAR": "Paraguay",
  "PER": "Peru",
  "PHI": "Philippines",
  "PLE": "Palestine",
  "PLW": "Palau",
  "PNG": "Papua New Guinea",
  "POL": "Poland",
  "POR": "Portugal",
  "PRK": "North Korea",
  "PUR": "Puerto Rico",
  "QAT": "Qatar",
  "ROU": "Romania",
  "RSA": "South Africa",
  "RUS": "Russia",
  "RWA": "Rwanda",
  "SAM": "Samoa",
  "SEN": "Senegal",
  "SEY": "Seychelles",
  "SGP": "Singapore",
  "SKN": "Saint Kitts and Nevis",
  "SLE": "Sierra Leone",
  "SLO": "Slovenia",
  "SMR": "San Marino",
  "SOL": "Solomon Islands",
  "SOM": "Somalia",
  "SRB": "Serbia",
  "SRI": "Sri Lanka",
  "SSD": "South Sudan",
  "STP": "Sao Tome and Principe",
  "SUD": "Sudan",
  "SUI": "Switzerland",
  "SUR": "Suriname",
  "SVK": "Slovakia",
  "SWE": "Sweden",
  "SWZ": "Eswatini",
  "SYR": "Syria",
  "TAN": "Tanzania",
  "TGA": "Tonga",
  "THA": "Thailand",
  "TJK": "Tajikistan",
  "TKM": "Turkmenistan",
  "TLS": "Timor-Leste",
  "TOG": "Togo",
  "TPE": "Chinese Taipei",
  "TTO": "Trinidad and Tobago",
  "TUN": "Tunisia",
  "TUR": "Turkey",
  "TUV": "Tuvalu",
  "UAE": "United Arab Emirates",
  "UGA": "Uganda",
  "UKR": "Ukraine",
  "URU": "Uruguay",
  "USA": "United States",
  "UZB": "Uzbekistan",
  "VAN": "Vanuatu",
  "VEN": "Venezuela",
  "VNM": "Vietnam",
  "VIE": "Vietnam",
  "VIN": "Saint Vincent and the Grenadines",
  "YEM": "Yemen",
  "ZAM": "Zambia",
  "RHO": "Zimbabwe",
  "ZIM": "Zimbabwe"
}


medal_data = pd.read_csv("2025_Problem_C_Data/summerOly_medal_counts.csv")
athlete_data = pd.read_csv("2025_Problem_C_Data/summerOly_athletes.csv")

medal_data["NOC"] = medal_data["NOC"].str.strip()
athlete_data["NOC"] = athlete_data["NOC"].str.strip()
athlete_data["Year"] = athlete_data["Year"].astype(int)
athlete_data["NOC"] = athlete_data["NOC"].replace(noc_to_country)

medal_data["HasMedal"] = medal_data[["Gold", "Silver", "Bronze"]].sum(axis=1) > 0

first_medal_year = medal_data[medal_data["HasMedal"]].groupby("NOC")["Year"].min()

first_medal_df = first_medal_year.reset_index()
first_medal_df.columns = ["NOC", "FirstMedalYear"]


def calc_oly_before_medal(row):
    noc = row["NOC"]
    first_year = row["FirstMedalYear"]
    # Filter athlete data for the specific country and years before their first medal
    participated_years = athlete_data[athlete_data["Team"] == noc]
    
    participated_before = participated_years[participated_years["Year"] < first_year]["Year"].drop_duplicates()

    return len(participated_before)

#code for printing out the table of people who have played or not before
# first_medal_df["OlympicsBeforeMedal"] = first_medal_df.apply(calc_oly_before_medal, axis = 1)
# print(first_medal_df.sort_values("OlympicsBeforeMedal"))


participating_countries = athlete_data["NOC"].unique()
countries_with_medals = first_medal_df["NOC"].unique()
not_real_countries = {"URS", "ANZ", "WIF", "AIN", "YUG", "EUN", "FRG", "SCG", "AHO", "CRT", "ROT", "BOH", "NBO", "YAR", "TCH", "EOR", "IOA", "SAA", "GDR", "YMD", "NFL", "UAR", "UNK"}
countries_without_medals = set(participating_countries) - set(countries_with_medals) - set(not_real_countries)

athlete_data = athlete_data[~athlete_data["NOC"].isin(not_real_countries)]

#see how many olys they've already done
def calc_olys_participated_in(noc):
    participating_years = athlete_data[athlete_data["NOC"] == noc]["Year"].drop_duplicates()
    return list(participating_years)

countries_without_medals_participation = {
    noc: calc_olys_participated_in(noc) for noc in countries_without_medals
}


no_medals_df = pd.DataFrame({
    "NOC": list(countries_without_medals_participation.keys()),
    "ParticipationYears": list(countries_without_medals_participation.values())
})


no_medals_df["TotalOlympicsParticipated"] = no_medals_df["ParticipationYears"].apply(len)
total_countries_without_medals = len(countries_without_medals)

# code for showing plot of total oly vs number countries without medals
# plt.figure(figsize=(10, 6))
# plt.hist(no_medals_df["TotalOlympicsParticipated"], bins=range(0, no_medals_df["TotalOlympicsParticipated"].max() + 2), edgecolor="black", color='lightcoral')
# plt.xlabel("Total Olympics Participated")
# plt.ylabel("Number of Countries")
# plt.title("Olympics Participation for Countries Without Medals")
# plt.xticks(range(0, no_medals_df["TotalOlympicsParticipated"].max() + 1))
# plt.tight_layout()
# plt.show()



# # Code for showing the bar graph of number of olympics before first medal vs num countries that participated
# first_medal_df_sorted = first_medal_df.sort_values("OlympicsBeforeMedal", ascending=False)

# plt.figure(figsize=(10, 6))
# plt.hist(first_medal_df["OlympicsBeforeMedal"], bins=range(0, first_medal_df["OlympicsBeforeMedal"].max() + 2), edgecolor="black", color='skyblue')

# plt.xlabel("Number of Olympics Before First Medal")
# plt.ylabel("Number of Countries")
# plt.title("Distribution of Olympics Before First Medal (Number of Countries)")
# plt.xticks(range(0, first_medal_df["OlympicsBeforeMedal"].max() + 1))
# plt.tight_layout()

# plt.show()

# we want to find the number of countries that win a medal for the first time by olympic years - and what proportion that is out of all the participating countries that year

first_time_medals = athlete_data[athlete_data["Year"].isin(first_medal_year.values) & (athlete_data["Year"] == athlete_data["NOC"].map(first_medal_year))]
first_time_winners_by_year = first_time_medals.groupby("Year")["NOC"].nunique()

total_participating_by_year = athlete_data.groupby("Year")["NOC"].nunique()
proportion_first_time_winners = (first_time_winners_by_year / total_participating_by_year).fillna(0)

# plt.figure(figsize=(10, 6))
# plt.plot(first_time_winners_by_year.index, first_time_winners_by_year.values, marker="o", color="skyblue")
# for year, proportion in zip(first_time_winners_by_year.index, first_time_winners_by_year.values):
#     plt.text(year, proportion, f"{proportion:.2f}", fontsize=9, ha="center", va="bottom", color="black")
# plt.title("Proportion of First-Time Medal-Winning Countries per Olympics")
# plt.xlabel("Year")
# plt.ylabel("Proportion")
# plt.grid(True)
# plt.show()

# Filter the data for years 1960 onwards
filtered_proportion = proportion_first_time_winners[proportion_first_time_winners.index > 1996]


# plt.figure(figsize=(10, 6))
# plt.plot(filtered_proportion.index, filtered_proportion.values, 
#          marker="o", color="skyblue", label="Proportion of First-Time Winners")
# for year, proportion in zip(filtered_proportion.index, filtered_proportion.values):
#     plt.text(year, proportion, f"{proportion:.2f}", 
#              fontsize=9, ha="center", va="bottom", color="black")
# plt.title("Proportion of First-Time Medal-Winning Countries per Olympics (1996 Onwards)")
# plt.xlabel("Year")
# plt.ylabel("Proportion")
# plt.grid(True)
# plt.legend()
# plt.show()

# years = filtered_proportion.index.values.reshape(-1, 1)  # Years as feature (X)
# proportions = filtered_proportion.values  # Proportions as target (y)

# Fit a linear regression model
# model = LinearRegression()
# model.fit(years, proportions)
# r_squared = model.score(years, proportions)
# print(f"R^2: {r_squared:.3f}")
# # Predict for 2028
# year_2028 = np.array([[2028]])
# pred_2028 = model.predict(year_2028)

# # Plot the data and regression line
# plt.figure(figsize=(10, 6))
# plt.scatter(years, proportions, color='skyblue', label='Observed Data', marker="o")
# plt.plot(years, model.predict(years), color='orange', label='Linear Regression Line')
# plt.scatter(2028, pred_2028, color='red', label=f'Prediction for 2028: {pred_2028[0]:.3f}', zorder=5)

# # Add labels and title
# plt.title("Proportion of First-Time Medal-Winning Countries per Olympics (Linear Regression)")
# plt.xlabel("Year")
# plt.ylabel("Proportion")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Reshape data for polynomial regression
# years = filtered_proportion.index.values.reshape(-1, 1)
# proportions = filtered_proportion.values

# # Apply polynomial transformation (degree 2 for quadratic)
# poly = PolynomialFeatures(degree=2)
# years_poly = poly.fit_transform(years)

# # Fit the model
# model = LinearRegression()
# model.fit(years_poly, proportions)


# # Make predictions
# predicted_proportions = model.predict(years_poly)
# years_new = np.array([[2028]])

# # Apply polynomial transformation
# years_new_poly = poly.transform(years_new)

# # Predict for 2028
# predicted_2028 = model.predict(years_new_poly)
# r_squared = model.score(years_poly, proportions)
# print(f"R^2: {r_squared:.3f}")
# print(f"Predicted proportion for 2028: {predicted_2028[0]}")

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(filtered_proportion.index, filtered_proportion.values, marker="o", color="skyblue", label="Actual")
# plt.plot(filtered_proportion.index, predicted_proportions, color="red", label="Polynomial Fit")
# for year, proportion in zip(filtered_proportion.index, filtered_proportion.values):
#     plt.text(year, proportion, f"{proportion:.2f}", fontsize=9, ha="center", va="bottom", color="black")

# plt.title("Proportion of First-Time Medal-Winning Countries per Olympics (Polynomial Fit)")
# plt.xlabel("Year")
# plt.ylabel("Proportion")
# plt.grid(True)
# plt.legend()
# plt.show()
# Calculate the change in events for each NOC

def calc_change_in_events(noc):
    country_data = athlete_data[athlete_data["NOC"] == noc]
    events_by_year = country_data.groupby("Year")["Event"].nunique()
    return events_by_year.diff().fillna(0).tolist()

no_medals_df["ChangeInEvents"] = no_medals_df["NOC"].apply(calc_change_in_events)

# Create the training dataset
# Perform the merge
training_data = pd.merge(
    athlete_data,
    medal_data[['NOC', 'Year', 'HasMedal']],
    on=['NOC', 'Year'],
    how='left'
)

# Check the shape of the DataFrame to confirm it has rows
print(f"Training Data Shape: {training_data.shape}")
# Print the column names of the training_data DataFrame
print("Columns in training_data:")
print(training_data.columns)

# Ensure 'no_medals_df' has the correct structure for exploding
# Let's print and check 'no_medals_df' before exploding
print(no_medals_df.head())
# Check for missing values in the HasMedal column
training_data["HasMedal"] = training_data["HasMedal"].fillna(0).astype(int)

# Fill missing values in 'HasMedal' with 0
training_data['HasWonMedal'] = training_data['HasMedal']
print(training_data[['HasMedal', 'HasWonMedal']].head())
# Ensure 'ParticipationYears' and 'ChangeInEvents' are in list format
no_medals_df["ParticipationYears"] = no_medals_df["ParticipationYears"].apply(lambda x: [x] if not isinstance(x, list) else x)
no_medals_df["ChangeInEvents"] = no_medals_df["ChangeInEvents"].apply(lambda x: [x] if not isinstance(x, list) else x)
print(training_data[['HasMedal', 'HasWonMedal']].head())


expanded_rows = []

# Iterate over each row in the dataframe
for _, row in no_medals_df.iterrows():
    # Get the values for ParticipationYears and ChangeInEvents
    participation_years = row["ParticipationYears"]
    change_in_events = row["ChangeInEvents"]
    
    # Check if they are lists (if not, wrap them into lists)
    if not isinstance(participation_years, list):
        participation_years = [participation_years]
    if not isinstance(change_in_events, list):
        change_in_events = [change_in_events]
    
    # Iterate through the lists and create a new row for each entry
    for year, event_change in zip(participation_years, change_in_events):
        new_row = row.copy()  # Copy the current row to preserve the other columns
        new_row["ParticipationYears"] = year  # Set the 'ParticipationYears' for this row
        new_row["ChangeInEvents"] = event_change  # Set the 'ChangeInEvents' for this row
        expanded_rows.append(new_row)  # Add the new row to the list

# Create a new DataFrame from the expanded rows
expanded_df = pd.DataFrame(expanded_rows)

# Reset index to avoid index alignment issues
expanded_df = expanded_df.reset_index(drop=True)

# Now, ensure the 'Year' column matches 'ParticipationYears' (or any other adjustments)
expanded_df["Year"] = expanded_df["ParticipationYears"]

# Ensure that 'ChangeInEvents' is a float
expanded_df["ChangeInEvents"] = expanded_df["ChangeInEvents"].astype(float)

# Now you can assign columns safely between the DataFrames
# If you want to assign values from expanded_df to training_data, reset the index of training_data as well
training_data = training_data.reset_index(drop=True)

# Assign columns from expanded_df to training_data
training_data["Year"] = expanded_df["Year"]
training_data["ChangeInEvents"] = expanded_df["ChangeInEvents"]

# Ensure that the 'HasMedal' column is still intact
print(training_data[['HasMedal', 'Year', 'ChangeInEvents']].head())

# Check the columns of training_data to ensure HasMedal exists
print(training_data.columns)

# If the merge was successful and HasMedal exists, fill NaN values and convert to integer
if "HasMedal" in training_data.columns:
    training_data["HasWonMedal"] = training_data["HasMedal"].fillna(0).astype(int)
else:
    print("HasMedal column is missing in training_data")
training_data["TotalOlympicsParticipated"] = training_data.groupby("NOC")["Year"].transform("nunique")

# Features and target variable
X = training_data[["TotalOlympicsParticipated", "ChangeInEvents"]]
y = training_data["HasWonMedal"]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train the model
logistic_model = LogisticRegression(class_weight='balanced')
logistic_model.fit(X_train, y_train)

# Evaluate the model
y_pred = logistic_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Prepare the 2028 dataset
countries_2028 = no_medals_df.copy()
countries_2028["TotalOlympicsParticipated"] += 1  # Add one for 2028
# Group events by year and count unique events
events_by_year = athlete_data.groupby("Year")["Event"].nunique()

# Filter for years 2012 and onwards
events_since_2012 = events_by_year[events_by_year.index >= 2012]

# Calculate the change in events between consecutive Olympics
change_in_events = events_since_2012.diff()

# Display the changes
print("Number of events by year since 2012:")
print(events_since_2012)
print("\nChange in events since 2012:")
print(change_in_events)

# Estimate the change for 2028
# Option 1: Use the average change since 2012
avg_change_since_2012 = change_in_events.mean()
print(f"\nAverage change in events since 2012: {avg_change_since_2012:.2f}")

# Option 2: Use the most recent change
recent_change = change_in_events.iloc[-1]
print(f"Most recent change in events: {recent_change}")

# Choose one as the estimated change for 2028
estimated_change_2028 = avg_change_since_2012  # or recent_change
print(f"\nEstimated change in events for 2028: {estimated_change_2028:.2f}")

countries_2028["ChangeInEvents"] = avg_change_since_2012; 
X_2028 = countries_2028[["TotalOlympicsParticipated", "ChangeInEvents"]]

# Predict probabilities for 2028
probs_2028 = logistic_model.predict_proba(X_2028)[:, 1]
threshold = 0.75

countries_2028["PredictedMedalWin"] = (probs_2028 > threshold).astype(int)

# Count predicted first-time winners
predicted_first_time_winners = countries_2028["PredictedMedalWin"].sum()
print("Predicted number of first-time medal-winning countries in 2028:", predicted_first_time_winners)

# Plot probabilities
plt.figure(figsize=(10, 6))
plt.bar(countries_2028["NOC"], probs_2028, color='skyblue')
plt.axhline(threshold, color='red', linestyle='--', label='Threshold (0.5)')
plt.title("Predicted Probabilities of First-Time Medal Wins (2028)")
plt.xlabel("Country (NOC)")
plt.ylabel("Probability")
plt.xticks(rotation=90)
plt.legend()
plt.show()
