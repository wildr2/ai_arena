from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

class User:
	def __init__(self, username):
		self.username = username
		self.chr_submitted = False
		self.chr_desc = ""

class Game:
	def __init__(self):
		self.users = {}
		self.max_users = 3
		self.arena_desc = ""

	def get_user(self, username):
		return self.users.get(username, None)

	def add_get_user(self, username):
		if username and len(username) > 0:
			if username in self.users:
				return self.users[username]

			if len(self.users) < self.max_users:
				user = User(username)
				self.users[username] = user
				if self.get_ready_count() == self.max_users:
					self.arena_desc = "The crowd goes wild..."
				return user

		return None
	
	def get_ready_count(self):
		return len(self.users)

	def get_response_creation(self):
		return jsonify({
			"status": "creation",
			"ability_options": ["You are all powerful", "You can sing pretty well", "You can change hair color at will, but only when it is raining."],
			"weakness_options": ["You are allergic to peanuts", "You are blind"],
			"item_options": ["A pineapple", "A sturdy wooden shield"],
			"ability_pick_count": 1,
			"weakness_pick_count": 1,
			"item_pick_count": 1
		})
	
	def get_response_arena(self, user):
		ready = self.get_ready_count()
		user_desc = user.chr_desc if user else "You are a spectator."
		return jsonify({
			"status": "arena",
			"chr_desc": user_desc,
			"ready": ready,
			"max": self.max_users,
			"arena_desc": self.arena_desc
		})


game = Game()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/connect', methods=['POST'])
@cross_origin()
def connect():
	global game
	data = request.json
	username = data.get("username")
	user = game.add_get_user(username)

	if user and not user.chr_submitted:
		return game.get_response_creation()
	else:
		return game.get_response_arena(user)

@app.route('/create-character', methods=['POST'])
@cross_origin()
def create_character():
	global game
	data = request.json
	username = data.get("username")
	user = game.get_user(username)
	if user:
		user.chr_submitted = True
		user.chr_desc = f"You are {username}..."
	return game.get_response_arena(user)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)