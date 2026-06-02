# Correct PIN for the ATM login
correct_pin = "1234"

# Count how many wrong attempts the user makes
attempts = 0

# Allow the user to try while attempts are less than 3
while attempts < 3:
    # Ask the user to enter the PIN
    pin = input("Enter your PIN: ")

    # Check if the entered PIN is correct
    if pin == correct_pin:
        print("Access Granted")
        # Stop the loop if the PIN is correct
        break
    else:
        # Increase the attempt count after a wrong PIN
        attempts += 1
        print("Wrong PIN")

# If the user enters the wrong PIN 3 times, block the card
if attempts == 3:
    print("Card Blocked")
