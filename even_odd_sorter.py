# Function to separate even and odd numbers from a list
def sort_numbers(numbers):
    # Empty list to store even numbers
    even_numbers = []

    # Empty list to store odd numbers
    odd_numbers = []

    # Loop through each number in the given list
    for number in numbers:
        # If number is divisible by 2, it is even
        if number % 2 == 0:
            even_numbers.append(number)
        # Otherwise, it is odd
        else:
            odd_numbers.append(number)

    # Return both lists
    return even_numbers, odd_numbers


# Test the function with at least 10 numbers
numbers = [10, 15, 22, 33, 40, 51, 64, 77, 88, 99]

# Call the function
evens, odds = sort_numbers(numbers)

# Print the results
print("Even numbers:", evens)
print("Odd numbers:", odds)
