from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/coffee_model.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        gender = int(request.form.get("gender"))
        age = int(request.form.get("age"))

        data = pd.DataFrame({
            "gender": [gender],
            "age": [age]
        })

        prediction = model.predict(data)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
