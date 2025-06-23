from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)
data = None

@app.route("/", methods=["GET", "POST"])
def index():
    global data
    result = None
    plot_html = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file:
                data = pd.read_csv(file)
                result = data.head().to_html(classes="table table-striped")

        elif "clean_action" in request.form:
            action = request.form["clean_action"]
            if action == "Remove Duplicates":
                before = data.shape[0]
                data = data.drop_duplicates()
                after = data.shape[0]
                result = f"<p>Removed {before - after} duplicates.</p>" + data.head().to_html(classes="table table-striped")

            elif action == "Drop NAs":
                before = data.shape[0]
                data = data.dropna()
                after = data.shape[0]
                result = f"<p>Removed {before - after} rows with NAs.</p>" + data.head().to_html(classes="table table-striped")

        elif "model_action" in request.form:
            model_type = request.form["model_type"]
            target = request.form["target"]
            features = request.form.getlist("features")
            split = int(request.form["split"])
            
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split/100, random_state=42)
            
            if model_type == "Linear":
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                result = f"<p>Linear Regression MSE: {mse:.2f}</p>"

            elif model_type == "Logistic":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                result = f"<p>Logistic Regression Accuracy: {acc:.2f}</p>"

            elif model_type == "Tree":
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                result = f"<p>Decision Tree Accuracy: {acc:.2f}</p>"

        elif "plot_action" in request.form:
            plot_type = request.form["plot_type"]
            x = request.form["x_col"]
            y = request.form.get("y_col")

            if plot_type == "Bar":
                fig = px.bar(data, x=x, y=y)
            elif plot_type == "Scatter":
                fig = px.scatter(data, x=x, y=y)
            elif plot_type == "Line":
                fig = px.line(data, x=x, y=y)
            elif plot_type == "Histogram":
                fig = px.histogram(data, x=x)
            elif plot_type == "Pie":
                fig = px.pie(data, names=x)
            elif plot_type == "Box":
                fig = px.box(data, x=x, y=y)
            
            plot_html = pio.to_html(fig, full_html=False)

    return render_template("index.html", data=data, result=result, plot_html=plot_html)

if __name__ == "__main__":
    app.run(debug=True)
