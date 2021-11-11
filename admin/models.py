from flask import Flask, jsonify, request, session, redirect
from passlib.hash import pbkdf2_sha256
from app import db
import uuid


class Admin:

    def startSession(self,admin):
        del admin['password']
        session['logged_in'] = True
        session['admin'] = admin

        return jsonify(admin), 200

    def signup(self):
        # create the admin object
        admin = {
            "_id": uuid.uuid4().hex,
            "name": request.form.get('name'),
            "email": request.form.get('email'),
            "password": request.form.get('password'),
        }

        # encript the password
        admin['password'] = pbkdf2_sha256.encrypt(admin['password'])

        #Check for existing email address
        if db.users.find_one({"email":admin['email']}):
            return jsonify({"error":"Email address already in use"}),400

        if db.users.insert_one(admin):
            return self.startSession(admin)

        return jsonify({"error":"Signup failed"}), 400

    def signOut(self):
        session.clear()
        return redirect('/')

    def login(self):
        admin = db.users.find_one({
            "email" : request.form.get("email")
        })
        if admin and pbkdf2_sha256.verify(request.form.get('password'),admin['password']) :
            return self.startSession(admin)
            
        return jsonify({"Error":"Invalid login credentials"}), 401
