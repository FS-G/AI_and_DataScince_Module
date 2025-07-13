# Excel Lookup Functions Reference

## VLOOKUP
**Purpose**: Searches for a value in the first column of a table and returns a value in the same row from a specified column.

**Syntax**: `=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])`

**Parameters**:
- `lookup_value`: The value to search for
- `table_array`: The range containing the data
- `col_index_num`: Column number to return value from (1 = first column)
- `range_lookup`: TRUE (approximate match) or FALSE (exact match)

**Example**: `=VLOOKUP(A2, B:D, 3, FALSE)`

## HLOOKUP
**Purpose**: Searches for a value in the top row of a table and returns a value in the same column from a specified row.

**Syntax**: `=HLOOKUP(lookup_value, table_array, row_index_num, [range_lookup])`

**Example**: `=HLOOKUP(A2, B1:F5, 3, FALSE)`

## INDEX & MATCH
**Purpose**: More flexible alternative to VLOOKUP that can search in any direction.

**Syntax**: `=INDEX(return_array, MATCH(lookup_value, lookup_array, [match_type]))`

**Parameters**:
- `return_array`: The column/row to return values from
- `lookup_value`: The value to search for
- `lookup_array`: The column/row to search in
- `match_type`: 0 (exact match), 1 (less than), -1 (greater than)

**Example**: `=INDEX(C:C, MATCH(A2, B:B, 0))`

## XLOOKUP (Excel 365/2021)
**Purpose**: Modern replacement for VLOOKUP with more features and flexibility.

**Syntax**: `=XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found], [match_mode], [search_mode])`

**Example**: `=XLOOKUP(A2, B:B, C:C, "Not Found")`

## LOOKUP
**Purpose**: Finds the largest value less than or equal to the lookup value.

**Syntax**: `=LOOKUP(lookup_value, lookup_vector, [result_vector])`

**Example**: `=LOOKUP(A2, B:B, C:C)`

## Common Tips

### Error Handling
- Use `IFERROR()` to handle lookup errors: `=IFERROR(VLOOKUP(A2, B:D, 3, FALSE), "Not Found")`
- Use `IFNA()` specifically for #N/A errors: `=IFNA(VLOOKUP(A2, B:D, 3, FALSE), "Not Found")`

### Best Practices
- Always use FALSE for exact matches unless you specifically need approximate matching
- Consider using absolute references ($) for table arrays: `$B$2:$D$100`
- INDEX/MATCH is generally faster than VLOOKUP for large datasets
- XLOOKUP can search from bottom to top and handles arrays better than VLOOKUP

### Common Errors
- **#N/A**: Value not found (use exact match or check data)
- **#REF!**: Invalid column/row reference
- **#VALUE!**: Wrong data type in lookup value
- **#NAME?**: Function name misspelled

## Quick Reference Table

| Function | Direction | Flexibility | Excel Version |
|----------|-----------|-------------|---------------|
| VLOOKUP  | Left to Right | Limited | All |
| HLOOKUP  | Top to Bottom | Limited | All |
| INDEX/MATCH | Any | High | All |
| XLOOKUP | Any | Very High | 365/2021+ |
| LOOKUP | Any | Medium | All |

## Sample Data Tables for Practice

### Table 1: Employee Database (A1:D6)
| Employee ID | Name | Department | Salary |
|-------------|------|------------|---------|
| 101 | John Smith | Sales | 50000 |
| 102 | Sarah Johnson | Marketing | 55000 |
| 103 | Mike Davis | IT | 62000 |
| 104 | Lisa Chen | HR | 48000 |
| 105 | Tom Wilson | Sales | 52000 |

### Table 2: Product Inventory (F1:H8)
| Product Code | Product Name | Price |
|--------------|--------------|--------|
| P001 | Laptop | 999.99 |
| P002 | Mouse | 25.50 |
| P003 | Keyboard | 75.00 |
| P004 | Monitor | 299.99 |
| P005 | Printer | 199.99 |
| P006 | Tablet | 399.99 |
| P007 | Headphones | 89.99 |

### Table 3: Sales Data by Quarter (J1:N6)
| Salesperson | Q1 | Q2 | Q3 | Q4 |
|-------------|----|----|----|----|
| Alice | 15000 | 18000 | 22000 | 25000 |
| Bob | 12000 | 16000 | 19000 | 21000 |
| Carol | 18000 | 20000 | 24000 | 28000 |
| David | 14000 | 17000 | 20000 | 23000 |
| Emma | 16000 | 19000 | 23000 | 26000 |

## Practical Examples with Sample Data

### Example 1: Employee Lookup (Using Table 1)
**Task**: Find employee name for ID 103
**Formula**: `=VLOOKUP(103, A2:D6, 2, FALSE)`
**Result**: "Mike Davis"

**Task**: Find department for employee ID 102
**Formula**: `=VLOOKUP(102, A2:D6, 3, FALSE)`
**Result**: "Marketing"

### Example 2: Product Price Lookup (Using Table 2)
**Task**: Find price for product code "P004"
**Formula**: `=VLOOKUP("P004", F2:H8, 3, FALSE)`
**Result**: 299.99

**Task**: Find product name for code "P001"
**Formula**: `=VLOOKUP("P001", F2:H8, 2, FALSE)`
**Result**: "Laptop"

### Example 3: INDEX/MATCH Alternative
**Task**: Find salary for employee "Lisa Chen"
**Formula**: `=INDEX(D2:D6, MATCH("Lisa Chen", B2:B6, 0))`
**Result**: 48000

### Example 4: HLOOKUP with Sales Data (Using Table 3)
**Task**: Find Q3 sales for Bob
**Formula**: `=HLOOKUP("Bob", J1:N6, 4, FALSE)`
**Result**: 19000

### Example 5: Two-Way Lookup
**Task**: Find Q2 sales for Carol
**Formula**: `=INDEX(K2:N6, MATCH("Carol", J2:J6, 0), MATCH("Q2", K1:N1, 0))`
**Result**: 20000

### Example 6: Error Handling
**Task**: Look up employee ID 999 (doesn't exist)
**Formula**: `=IFERROR(VLOOKUP(999, A2:D6, 2, FALSE), "Employee not found")`
**Result**: "Employee not found"

### Example 7: XLOOKUP (Excel 365/2021)
**Task**: Find department for employee ID 104
**Formula**: `=XLOOKUP(104, A2:A6, C2:C6, "Not Found")`
**Result**: "HR"

## Student Practice Exercises

### Exercise 1: Basic VLOOKUP
Create a lookup formula to find the salary of employee ID 105.

### Exercise 2: Product Information
Write a formula to find the product name for code "P007".

### Exercise 3: Department Analysis
Use INDEX/MATCH to find the salary of the first employee in the "Sales" department.

### Exercise 4: Quarterly Performance
Create an HLOOKUP formula to find Alice's Q4 sales.

### Exercise 5: Error Prevention
Write a formula that looks up product "P999" and returns "Product not available" if not found.

### Exercise 6: Advanced Challenge
Create a formula that finds the highest salary in the IT department.

## Solutions

### Solution 1: `=VLOOKUP(105, A2:D6, 4, FALSE)` → 52000
### Solution 2: `=VLOOKUP("P007", F2:H8, 2, FALSE)` → "Headphones"
### Solution 3: `=INDEX(D2:D6, MATCH("Sales", C2:C6, 0))` → 50000
### Solution 4: `=HLOOKUP("Alice", J1:N6, 5, FALSE)` → 25000
### Solution 5: `=IFERROR(VLOOKUP("P999", F2:H8, 2, FALSE), "Product not available")`
### Solution 6: `=INDEX(D2:D6, MATCH("IT", C2:C6, 0))` → 62000