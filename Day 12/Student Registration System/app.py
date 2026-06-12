import sqlite3
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "students.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = "student-registration-demo-secret"


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            roll_number TEXT NOT NULL UNIQUE,
            department TEXT NOT NULL,
            year TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            gender TEXT NOT NULL,
            address TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    db.commit()


def insert_student(form_data):
    db = get_db()
    db.execute(
        """
        INSERT INTO students (
            student_name, roll_number, department, year,
            email, phone, gender, address
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            form_data["student_name"],
            form_data["roll_number"],
            form_data["department"],
            form_data["year"],
            form_data["email"],
            form_data["phone"],
            form_data["gender"],
            form_data["address"],
        ),
    )
    db.commit()


def get_students():
    db = get_db()
    return db.execute(
        """
        SELECT id, student_name, roll_number, department, year,
               email, phone, gender, address, created_at
        FROM students
        ORDER BY id DESC
        """
    ).fetchall()


@app.before_request
def ensure_database():
    init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        form_data = {
            "student_name": request.form.get("student_name", "").strip(),
            "roll_number": request.form.get("roll_number", "").strip(),
            "department": request.form.get("department", "").strip(),
            "year": request.form.get("year", "").strip(),
            "email": request.form.get("email", "").strip(),
            "phone": request.form.get("phone", "").strip(),
            "gender": request.form.get("gender", "").strip(),
            "address": request.form.get("address", "").strip(),
        }

        missing_fields = [label for label, value in form_data.items() if not value]
        if missing_fields:
            flash("Please complete every field before submitting.", "error")
            return render_template("register.html", form_data=form_data)

        try:
            insert_student(form_data)
            flash("Student registered successfully and saved to SQLite.", "success")
            return redirect(url_for("student_records"))
        except sqlite3.IntegrityError:
            flash("A student with this roll number already exists.", "error")
            return render_template("register.html", form_data=form_data)

    return render_template("register.html", form_data={})


@app.route("/students")
def student_records():
    return render_template("students.html", students=get_students())


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True)
