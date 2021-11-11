from flask import Flask, render_template, session, redirect
from app import app
from admin.models import Admin

@app.route("/admin/user/")
def user():
    return render_template('admin/user.html')

@app.route("/admin/absensi/")
def absensi():
    return render_template('admin/absensi.html')

@app.route("/admin/report/")
def report():
    return render_template('admin/report.html')

@app.route('/admin/signup', methods=['POST'])
def signup():
    return Admin().signup()

@app.route('/admin/signout')
def signout():
    return Admin().signOut()

@app.route('/admin/signin', methods=['POST'])
def signin():
    return Admin().login()