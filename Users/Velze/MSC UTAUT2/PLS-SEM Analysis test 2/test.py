# UTAUT2 GameFi Data Cleaning Script
# Run this script locally with your Excel file named "dataset-utaut2.xlsx"

import pandas as pd
import numpy as np
import os


def clean_utaut2_data(file_path="dataset-utaut2.xlsx"):
    """
    Clean UTAUT2 GameFi dataset for PLS-SEM analysis
    """
    print("🚀 Starting UTAUT2 Data Cleaning Process...")

    # Step 1: Load the Excel file
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # First sheet
        print(f"✅ File loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

    # Step 2: Identify data rows vs questionnaire text rows
    # Based on analysis: rows with questionnaire text start around row 533
    # Let's find where actual responses end and questionnaire text begins

    # Check last few rows to identify cut-off point
    print("\n🔍 Identifying data vs text rows...")
    last_10_rows = df.tail(10)
    print("Sample of last rows to identify cut-off:")
    print(last_10_rows.iloc[:, 0].tolist())  # Show first column of last 10 rows

    # Remove questionnaire text rows (typically the last 3-5 rows)
    # Keep only rows that contain actual Likert responses
    data_rows = []
    for idx, row in df.iterrows():
        first_col_val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        # Check if first column contains actual Likert response
        if any(likert_term in first_col_val for likert_term in
               ['Strongly Agree', 'Agree', 'Neither', 'Disagree', 'Strongly Disagree']):
            data_rows.append(idx)

    # Keep only actual data rows
    df_clean = df.iloc[data_rows].copy()
    print(f"✅ Filtered to {len(df_clean)} actual data rows")

    # Step 3: Fix column names
    print("\n🔧 Cleaning column names...")

    # Create proper column names based on the structure we identified
    likert_columns = [
        'PE1', 'PE2', 'PE3', 'PE4', 'PE5',  # Performance Expectancy (5 items)
        'EE1', 'EE2', 'EE3', 'EE4',  # Effort Expectancy (4 items)
        'SI1', 'SI2', 'SI3',  # Social Influence (3 items)
        'FC1', 'FC2', 'FC3', 'FC4',  # Facilitating Conditions (4 items)
        'HM1', 'HM2', 'HM3', 'HM4',  # Hedonic Motivation (4 items)
        'PV1', 'PV2', 'PV3',  # Price Value (3 items)
        'HB1', 'HB2', 'HB3', 'HB4',  # Habit (4 items)
        'BI1', 'BI2', 'BI3',  # Behavioral Intention (3 items)
        'EM1', 'EM2', 'EM3',  # Economic Motivation (3 items)
        'RP1', 'RP2', 'RP3', 'RP4',  # Risk Perception (4 items)
        'TT1', 'TT2', 'TT3',  # Trust in Technology (3 items)
        'RC1', 'RC2',  # Regulatory Compliance (2 items)
    ]

    # UB items are actually behavioral usage questions (not traditional Likert)
    behavioral_columns = [
        'UB1_UseFreq',  # How often do you currently use GameFi platforms
        'UB2_WeeklyHours',  # Hours per week on GameFi platforms
    ]

    demographic_columns = [
        'Age',  # Age Group
        'Gender',  # Gender
        'Education',  # Highest Level of Education
        'Income',  # Monthly Income (USD equivalent)
        'GameFiExp',  # GameFi Experience
        'Location'  # Geographic Location/Region
    ]

    # Assign new column names (40 Likert + 2 behavioral + 6 demographics = 48 total)
    # Note: Adjusting for actual 50 columns in your file
    new_columns = likert_columns + behavioral_columns + demographic_columns

    if len(df_clean.columns) == len(new_columns):
        df_clean.columns = new_columns
        print("✅ Column names updated successfully")
    else:
        print(f"⚠️  Column mismatch: Expected {len(new_columns)}, got {len(df_clean.columns)}")
        # Use original columns if mismatch

    # Step 4: Convert Likert scale responses to numeric (1-5)
    print("\n🔢 Converting Likert responses to numeric scale...")

    # Define conversion mapping
    likert_mapping = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neither Agree nor Disagree': 3,
        'Neither Agree/Disagree': 3,
        'Neither': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }

    # Convert only Likert scale columns (first 42 columns)
    likert_cols = df_clean.columns[:42]  # First 42 are Likert items (including RC1, RC2)

    conversion_stats = {'converted': 0, 'errors': 0}

    for col in likert_cols:
        original_values = df_clean[col].value_counts()
        print(f"\nConverting {col}:")
        print(f"  Original values: {list(original_values.index[:3])}...")  # Show first 3 unique values

        # Apply mapping
        df_clean[col] = df_clean[col].map(likert_mapping)

        # Check for unmapped values (NaN after mapping)
        unmapped = df_clean[col].isna().sum()
        if unmapped > 0:
            conversion_stats['errors'] += unmapped
            print(f"  ⚠️  {unmapped} values couldn't be converted")
        else:
            conversion_stats['converted'] += 1
            print(f"  ✅ Converted successfully")

    print(f"\n📊 Conversion Summary:")
    print(f"  Successfully converted: {conversion_stats['converted']} columns")
    print(f"  Conversion errors: {conversion_stats['errors']} values")

    # Step 5: Handle demographic variables
    print("\n👥 Processing demographic variables...")

    # For demographics + behavioral items, we'll keep them as categorical for now
    # but create a summary to understand the coding
    non_likert_cols = df_clean.columns[42:]  # Last 8 columns are behavioral + demographics

    for col in non_likert_cols:
        unique_vals = df_clean[col].value_counts()
        print(f"\n{col}: {len(unique_vals)} unique values")
        print(f"  Top 3: {list(unique_vals.head(3).index)}")

    # Step 6: Data quality checks
    print("\n🔍 Performing data quality checks...")

    # Check for missing values in Likert items
    missing_likert = df_clean[likert_cols].isnull().sum()
    total_missing = missing_likert.sum()

    if total_missing > 0:
        print(f"⚠️  Found {total_missing} missing values in Likert items:")
        missing_cols = missing_likert[missing_likert > 0]
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing")
    else:
        print("✅ No missing values in Likert items")

    # Check Likert scale range (should be 1-5)
    print("\n📊 Likert scale range check:")
    for col in likert_cols[:5]:  # Check first 5 columns as sample
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        print(f"  {col}: range {min_val}-{max_val}")

    # Step 7: Create construct means for initial validation
    print("\n🧮 Computing construct means...")

    # Define construct groupings (only Likert scale constructs)
    constructs = {
        'PE': ['PE1', 'PE2', 'PE3', 'PE4', 'PE5'],
        'EE': ['EE1', 'EE2', 'EE3', 'EE4'],
        'SI': ['SI1', 'SI2', 'SI3'],
        'FC': ['FC1', 'FC2', 'FC3', 'FC4'],
        'HM': ['HM1', 'HM2', 'HM3', 'HM4'],
        'PV': ['PV1', 'PV2', 'PV3'],
        'HB': ['HB1', 'HB2', 'HB3', 'HB4'],
        'BI': ['BI1', 'BI2', 'BI3'],
        'EM': ['EM1', 'EM2', 'EM3'],
        'RP': ['RP1', 'RP2', 'RP3', 'RP4'],
        'TT': ['TT1', 'TT2', 'TT3'],
        'RC': ['RC1', 'RC2']
        # Note: UB is behavioral usage, not a traditional Likert construct
    }

    # Compute construct means
    construct_means = {}
    for construct, items in constructs.items():
        if all(item in df_clean.columns for item in items):
            construct_means[construct] = df_clean[items].mean(axis=1)
            mean_val = construct_means[construct].mean()
            std_val = construct_means[construct].std()
            print(f"  {construct}: M={mean_val:.2f}, SD={std_val:.2f}")
        else:
            print(f"  ⚠️  {construct}: Some items missing")

    # Step 8: Save cleaned data
    print("\n💾 Saving cleaned data...")

    # Save the cleaned dataset
    output_file = "utaut2_cleaned_data.xlsx"
    df_clean.to_excel(output_file, index=False)
    print(f"✅ Cleaned data saved to: {output_file}")

    # Save construct means for quick analysis
    construct_df = pd.DataFrame(construct_means)
    construct_df.to_excel("utaut2_construct_means.xlsx", index=False)
    print("✅ Construct means saved to: utaut2_construct_means.xlsx")

    # Step 9: Generate summary report
    print("\n📋 FINAL SUMMARY:")
    print(f"  📊 Dataset: {len(df_clean)} participants × {len(df_clean.columns)} variables")
    print(f"  🎯 Likert items: {len(likert_cols)} (converted to 1-5 scale)")
    print(f"  📊 Behavioral/Demographics: {len(non_likert_cols)} variables")
    print(f"  🏗️  Constructs: {len(constructs)} constructs defined")
    print(f"  💾 Output files: utaut2_cleaned_data.xlsx, utaut2_construct_means.xlsx")

    print("\n🎉 Data cleaning completed successfully!")
    print("\n📝 Next steps:")
    print("  1. Review the cleaned data file")
    print("  2. Check construct means for reasonableness")
    print("  3. Ready for PLS-SEM analysis!")

    return df_clean


# Run the cleaning function
if __name__ == "__main__":
    # Check if file exists
    file_name = "dataset-utaut2.xlsx"
    if os.path.exists(file_name):
        cleaned_data = clean_utaut2_data(file_name)
    else:
        print(f"❌ File '{file_name}' not found in current directory")
        print("📁 Current directory contents:")
        print([f for f in os.listdir('.') if f.endswith('.xlsx')])
        print("\n💡 Make sure your Excel file is named 'dataset-utaut2.xlsx'")
        print("   Or modify the file_name variable in this script")