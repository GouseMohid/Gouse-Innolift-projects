# Function to return a grade based on the student's marks
def get_grade(marks):
    # Marks 90 and above get grade A
    if marks >= 90:
        return "A"
    # Marks 75 to 89 get grade B
    elif marks >= 75:
        return "B"
    # Marks 60 to 74 get grade C
    elif marks >= 60:
        return "C"
    # Marks 45 to 59 get grade D
    elif marks >= 45:
        return "D"
    # Marks below 45 get grade F
    else:
        return "F"


# Loop runs for 5 students
for student in range(1, 6):
    # Take marks as input from the user
    marks = float(input(f"Enter marks for student {student}: "))

    # Call the function to get the grade
    grade = get_grade(marks)

    # Print the student's grade
    print(f"Student {student} grade: {grade}")
