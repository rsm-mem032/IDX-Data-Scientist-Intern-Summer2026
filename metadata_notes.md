# Metadata Notes

## Dataset Context

This project uses historical CRMLS sold property records to predict California residential single-family property close prices.

The modeling dataset should only include observations where:

- `PropertyType = "Residential"`
- `PropertySubType = "SingleFamilyResidence"`

Raw CRMLSSold files are stored locally only and are not uploaded to GitHub.

## Metadata Source

The field definitions are based on the Trestle Property Metadata document for the Property resource.

According to the metadata document, `ListingKey` is the primary key and uniquely identifies records in the Property resource.

## Key Columns

### Identifier

- `ListingKey`: Primary key that uniquely identifies each property record.

### Target Variable

- `ClosePrice`: The amount of money paid by the purchaser to the seller for the property under the agreement. This is the target variable for the prediction model.

### Required Filters

- `PropertyType`: General property category. This project only keeps records where `PropertyType = "Residential"`.
- `PropertySubType`: Specific property category. This project only keeps records where `PropertySubType = "SingleFamilyResidence"`.

### Date Field

- `CloseDate`: The date when the purchase agreement was fulfilled. For this project, this field will be used to understand transaction timing and support time-based train/test splitting.

### Core Property Features

- `LivingArea`: The living area of the property. This will be used as one of the main size-related predictors.
- `BedroomsTotal`: The total number of bedrooms in the dwelling.
- `BathroomsTotalInteger`: The simple sum of the number of bathrooms. For example, a property with two full bathrooms and one half bathroom has a Bathrooms Total Integer of 3.
- `LotSizeSquareFeet`: Lot size of the property measured in square feet.
- `YearBuilt`: The year the property was built. This can be used to calculate property age.

### Location Features

- `City`: The city in the listing address.
- `PostalCode`: Postal code of the property address.
- `CountyOrParish`: County, parish, or other regional authority.
- `Latitude`: Latitude coordinate of the property.
- `Longitude`: Longitude coordinate of the property.

## Initial Modeling Considerations

The first modeling dataset should prioritize fields that are likely to be available for most properties and directly related to price:

- `ClosePrice`
- `CloseDate`
- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`
- `YearBuilt`
- `City`
- `PostalCode`
- `CountyOrParish`
- `Latitude`
- `Longitude`
- `PropertyType`
- `PropertySubType`

Potential fields such as agent names, office names, phone numbers, emails, URLs, and other personally identifiable or operational fields should not be used in the initial predictive model.

## Week 1 Dataset Access Confirmation

Dataset access has been confirmed through FileZilla FTP.

Downloaded data:

- Approximately seven months of `CRMLSSold` files from `/raw/California`
- `Trestle Property MetaData.pdf` from `/resources`

The downloaded raw data files are stored locally in the `data/` directory and excluded from GitHub using `.gitignore`.
