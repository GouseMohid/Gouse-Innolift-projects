# Function to print the multiplication table of a number
def times_table(n):
    # Loop from 1 to 10
    for i in range(1, 11):
        # Print the multiplication result
        print(f"{n} x {i} = {n * i}")


# Print the table of 2
times_table(2)

# Print a blank line between tables
print()

# Print the table of 5
times_table(5)

# Print a blank line between tables
print()

# Print the table of 9
times_table(9)
