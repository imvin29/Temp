from flask import Flask, render_template, request, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    asana_name = request.form['asana']
    return render_template('poses.html', asana=asana_name)

@app.route('/practice', methods=['POST'])
def practice():
    asana_name = request.form['asana']
    print("Received asana:", asana_name)
    if asana_name:
        subprocess.run(["python", "single_pose_interface.py", asana_name])
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)