from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import random

app = Flask(__name__)
app.secret_key = "secret123"

# ===== DATABASE =====
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+mysqlconnector://root:123456789@localhost/fraud"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ===== MODELS =====
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)   # ممكن يتكرر
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.TIMESTAMP, default=db.func.current_timestamp())

class Transaction(db.Model):
    __tablename__ = "transactions"
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float)
    transaction_type = db.Column(db.String(50))
    location = db.Column(db.String(50))
    device_type = db.Column(db.String(50))
    time_of_day = db.Column(db.String(50))
    previous_fraud = db.Column(db.Integer)
    transaction_speed = db.Column(db.Float)
    prediction = db.Column(db.Integer)
    fraud_probability = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at = db.Column(db.TIMESTAMP, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

# ===== Load ML model =====
model = joblib.load("ranfor.pk")

transaction_map = {"Purchase": 0, "Withdrawal": 1, "Transfer": 2}
location_map = {"Chicago": 0, "Houston": 1, "Los Angeles": 2, "Miami": 3, "New York": 4}
device_map = {"Mobile": 0, "Laptop": 1, "ATM": 2}
timeofday_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

# ===== ROUTES =====
@app.route("/")
def home():
    return redirect(url_for("login"))

# ===== REGISTER =====
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        # تحقق من email فقط
        if User.query.filter_by(email=email).first():
            flash("Email déjà utilisé", "danger")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        flash("Compte créé avec succès !", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ===== LOGIN =====
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash("Email ou mot de passe incorrect", "danger")
            return redirect(url_for("login"))

        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("transaction_page"))

    return render_template("login.html")

# ===== LOGOUT =====
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ===== Transaction page =====
@app.route("/transaction")
def transaction_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ===== Graph generation =====
def generate_graphs(user_id):
    transactions = Transaction.query.filter_by(user_id=user_id).all()
    if not transactions:
        return []

    data_list = []
    for t in transactions:
        data_list.append({
            "Amount": t.amount,
            "TransactionType": t.transaction_type,
            "Location": t.location,
            "DeviceType": t.device_type,
            "TimeOfDay": t.time_of_day,
            "PreviousFraud": t.previous_fraud,
            "TransactionSpeed": t.transaction_speed,
            "Prediction": t.prediction
        })

    num_cols = ["Amount", "PreviousFraud", "TransactionSpeed", "Prediction"]
    text_cols = ["TransactionType", "Location", "DeviceType", "TimeOfDay"]
    columns = num_cols + text_cols
    graphs = []

    # Univariate
    uni_cols = random.sample(columns, 3)
    for col in uni_cols:
        if col in num_cols:
            fig = px.histogram(data_list, x=col, nbins=20, title=f"Distribution de {col}")
        else:
            fig = px.bar(data_list, x=col, title=f"Distribution de {col}")
        graphs.append(pio.to_json(fig))

    # Bivariate
    two_num = random.sample(num_cols, 2)
    fig = px.scatter(data_list, x=two_num[0], y=two_num[1], title=f"{two_num[0]} vs {two_num[1]}")
    graphs.append(pio.to_json(fig))

    two_text = random.sample(text_cols, 2)
    counts = {}
    for row in data_list:
        key = (row[two_text[0]], row[two_text[1]])
        counts[key] = counts.get(key, 0) + 1

    x_vals = [f"{k[0]} | {k[1]}" for k in counts.keys()]
    y_vals = list(counts.values())
    fig = go.Figure([go.Bar(x=x_vals, y=y_vals)])
    fig.update_layout(title=f"{two_text[0]} vs {two_text[1]}")
    graphs.append(pio.to_json(fig))

    num = random.choice(num_cols)
    txt = random.choice(text_cols)
    fig = px.box(data_list, x=txt, y=num, title=f"{txt} vs {num}")
    graphs.append(pio.to_json(fig))

    # Trivariate
    three_num = random.sample(num_cols, 3)
    fig = px.scatter_3d(data_list, x=three_num[0], y=three_num[1], z=three_num[2], color="Prediction",
                        title=f"{three_num[0]} vs {three_num[1]} vs {three_num[2]}")
    graphs.append(pio.to_json(fig))

    two_num = random.sample(num_cols, 2)
    one_text = random.choice(text_cols)
    fig = px.scatter_3d(data_list, x=two_num[0], y=two_num[1],
                        z=[hash(row[one_text]) % 10 for row in data_list],
                        color="Prediction",
                        title=f"{two_num[0]} vs {two_num[1]} vs {one_text}")
    graphs.append(pio.to_json(fig))

    one_num = random.choice(num_cols)
    two_text = random.sample(text_cols, 2)
    fig = px.scatter_3d(data_list,
                        x=[hash(row[two_text[0]]) % 10 for row in data_list],
                        y=[hash(row[two_text[1]]) % 10 for row in data_list],
                        z=[row[one_num] for row in data_list],
                        color="Prediction",
                        title=f"{two_text[0]} vs {two_text[1]} vs {one_num}")
    graphs.append(pio.to_json(fig))

    return graphs

@app.route("/analyse")
def analyse_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    graphsJSON = generate_graphs(session["user_id"])
    return render_template("analyse.html", graphsJSON=graphsJSON)

# API transactions
@app.route("/api/transactions")
def api_transactions():
    if "user_id" not in session:
        return jsonify([])
    user_id = session["user_id"]
    data = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.created_at.desc()).all()
    result = [{
        "Amount": t.amount,
        "TransactionType": t.transaction_type,
        "Location": t.location,
        "DeviceType": t.device_type,
        "TimeOfDay": t.time_of_day,
        "PreviousFraud": t.previous_fraud,
        "TransactionSpeed": t.transaction_speed,
        "Prediction": t.prediction
    } for t in data]
    return jsonify(result)

# ===== PREDICTION =====
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json()
    features = np.array([
        float(data["Amount"]),
        transaction_map[data["TransactionType"]],
        location_map[data["Location"]],
        device_map[data["DeviceType"]],
        timeofday_map[data["TimeOfDay"]],
        int(data["PreviousFraud"]),
        float(data["TransactionSpeed"])
    ]).reshape(1, -1)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    new_tr = Transaction(
        amount=float(data["Amount"]),
        transaction_type=data["TransactionType"],
        location=data["Location"],
        device_type=data["DeviceType"],
        time_of_day=data["TimeOfDay"],
        previous_fraud=int(data["PreviousFraud"]),
        transaction_speed=float(data["TransactionSpeed"]),
        prediction=int(pred),
        fraud_probability=float(prob),
        user_id=session["user_id"]
    )

    db.session.add(new_tr)
    db.session.commit()
    return jsonify({"prediction": int(pred), "fraud_probability": float(prob)})

@app.route("/transactions")
def show_transactions():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    data = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.created_at.desc()).all()
    result = [{
        "id": t.id,
        "amount": t.amount,
        "transaction_type": t.transaction_type,
        "location": t.location,
        "device_type": t.device_type,
        "time_of_day": t.time_of_day,
        "previous_fraud": t.previous_fraud,
        "transaction_speed": t.transaction_speed,
        "prediction": t.prediction,
        "fraud_probability": t.fraud_probability,
        "created_at": t.created_at.isoformat() if t.created_at else None
    } for t in data]
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
