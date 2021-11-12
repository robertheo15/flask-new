from flask import Flask, render_template, session, redirect
from app import app
from admin.models import Admin

@app.route("/admin/user/")
def adminUser():
    users = Admin().getAdmin()
    return render_template('admin/user.html', myUsers=users)

@app.route("/admin/user/create/")
def userCreate():
    return render_template('admin/create.html')   

@app.route('/admin/create-admin/', methods=['POST'])
def addAdmin():
    return Admin().addAdmin()     

@app.route('/admin/delete-admin/<id>', methods=['get'])
def deleteAdmin(id):
    Admin().deleteAdmin(id)
    return redirect("/admin/user/")
       
@app.route("/admin/absensi/")
def absensi():
    return render_template('admin/absensi.html')

@app.route("/admin/report/")
def report():
    reports = Admin().getReport()
    return render_template('admin/report.html', myReports=reports)

@app.route('/admin/signout')
def signout():
    return Admin().signOut()

@app.route('/admin/signin', methods=['POST'])
def signin():
    return Admin().login()