# Excel Lookup Functions - Student Reference Guide

## VLOOKUP (Vertical Lookup)
**Purpose**: Searches for a value in the first column of a table and returns a value in the same row from a specified column.

**Syntax**: `=VLOOKUP(lookup_value, table_array, col_index_num, range_lookup)`

**Parameters**:
- `lookup_value`: The value to search for
- `table_array`: The range containing the data
- `col_index_num`: Column number to return value from (1 = first column)
- `range_lookup`: TRUE (approximate match) or FALSE (exact match - recommended)

## HLOOKUP (Horizontal Lookup)
**Purpose**: Searches for a value in the top row of a table and returns a value in the same column from a specified row.

**Syntax**: `=HLOOKUP(lookup_value, table_array, row_index_num, range_lookup)`

**Parameters**:
- `lookup_value`: The value to search for
- `table_array`: The range containing the data
- `row_index_num`: Row number to return value from (1 = first row)
- `range_lookup`: TRUE (approximate match) or FALSE (exact match - recommended)

## Sample Data Tables for Practice

### Table 1: Student Database (A1:D6) - For VLOOKUP
| Student ID | Name | Grade | Age |
|------------|------|-------|-----|
| 101 | John Smith | A | 20 |
| 102 | Sarah Johnson | B | 19 |
| 103 | Mike Davis | A | 21 |
| 104 | Lisa Chen | C | 18 |
| 105 | Tom Wilson | B | 20 |

### Table 2: Monthly Sales (F1:K3) - For HLOOKUP
| Month | Jan | Feb | Mar | Apr | May |
|-------|-----|-----|-----|-----|-----|
| Sales | 5000 | 6000 | 7500 | 8000 | 9500 |
| Target | 4500 | 5500 | 7000 | 7500 | 9000 |

## VLOOKUP Examples with Table 1

### Example 1: Find Student Name
**Task**: Find the name of student with ID 103
**Formula**: `=VLOOKUP(103, A2:D6, 2, FALSE)`
**Result**: "Mike Davis"

**Explanation**: 
- 103 is the lookup value
- A2:D6 is the table range
- 2 means return value from column 2 (Name)
- FALSE means exact match

### Example 2: Find Student Grade
**Task**: Find the grade of student with ID 102
**Formula**: `=VLOOKUP(102, A2:D6, 3, FALSE)`
**Result**: "B"

### Example 3: Find Student Age
**Task**: Find the age of student with ID 105
**Formula**: `=VLOOKUP(105, A2:D6, 4, FALSE)`
**Result**: 20

## HLOOKUP Examples with Table 2

### Example 1: Find Sales Amount
**Task**: Find sales amount for March
**Formula**: `=HLOOKUP("Mar", F1:K3, 2, FALSE)`
**Result**: 7500

**Explanation**:
- "Mar" is the lookup value
- F1:K3 is the table range
- 2 means return value from row 2 (Sales)
- FALSE means exact match

### Example 2: Find Target Amount
**Task**: Find target amount for April
**Formula**: `=HLOOKUP("Apr", F1:K3, 3, FALSE)`
**Result**: 7500

### Example 3: Find January Sales
**Task**: Find sales amount for January
**Formula**: `=HLOOKUP("Jan", F1:K3, 2, FALSE)`
**Result**: 5000

## Student Practice Exercises

### VLOOKUP Exercises (Use Table 1)
1. Find the name of student with ID 104
2. Find the grade of student with ID 101
3. Find the age of student with ID 103

### HLOOKUP Exercises (Use Table 2)
1. Find the sales amount for February
2. Find the target amount for May
3. Find the sales amount for April

## Exercise Solutions

### VLOOKUP Solutions
1. `=VLOOKUP(104, A2:D6, 2, FALSE)` → "Lisa Chen"
2. `=VLOOKUP(101, A2:D6, 3, FALSE)` → "A"
3. `=VLOOKUP(103, A2:D6, 4, FALSE)` → 21

### HLOOKUP Solutions
1. `=HLOOKUP("Feb", F1:K3, 2, FALSE)` → 6000
2. `=HLOOKUP("May", F1:K3, 3, FALSE)` → 9000
3. `=HLOOKUP("Apr", F1:K3, 2, FALSE)` → 8000

## Common Tips for Students

### Remember:
- Always use FALSE for exact matches (most common)
- Column/Row numbers start from 1
- Make sure your lookup value exists in the first column (VLOOKUP) or first row (HLOOKUP)
- Check your table range includes all necessary data

### Common Errors:
- **#N/A**: Value not found - check spelling and data
- **#REF!**: Invalid column/row number - check your index number
- **#VALUE!**: Wrong data type - make sure numbers match numbers, text matches text