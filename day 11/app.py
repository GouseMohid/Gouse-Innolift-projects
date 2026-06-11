from flask import Flask, render_template, request


app = Flask(__name__)


students = [
    {"roll_no": 101, "name": "Arjun Kumar", "department": "IT", "year": "3rd", "email": "arjun101@citycollege.edu"},
    {"roll_no": 102, "name": "Priya Sharma", "department": "CSE", "year": "2nd", "email": "priya102@citycollege.edu"},
    {"roll_no": 103, "name": "Rahul Verma", "department": "ECE", "year": "1st", "email": "rahul103@citycollege.edu"},
    {"roll_no": 104, "name": "Sneha Reddy", "department": "EEE", "year": "4th", "email": "sneha104@citycollege.edu"},
    {"roll_no": 105, "name": "Vikram Singh", "department": "Mechanical", "year": "3rd", "email": "vikram105@citycollege.edu"},
    {"roll_no": 106, "name": "Meera Joshi", "department": "Civil", "year": "2nd", "email": "meera106@citycollege.edu"},
    {"roll_no": 107, "name": "Aditya Rao", "department": "CSE", "year": "1st", "email": "aditya107@citycollege.edu"},
    {"roll_no": 108, "name": "Kavya Nair", "department": "IT", "year": "4th", "email": "kavya108@citycollege.edu"},
    {"roll_no": 109, "name": "Farhan Ali", "department": "ECE", "year": "3rd", "email": "farhan109@citycollege.edu"},
    {"roll_no": 110, "name": "Ananya Das", "department": "AIML", "year": "2nd", "email": "ananya110@citycollege.edu"},
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    submitted = request.method == "POST"
    return render_template("register.html", submitted=submitted)


@app.route("/students")
def student_records():
    return render_template("students.html", students=students)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
