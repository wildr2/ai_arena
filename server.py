from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from server_lib import *

class User:
	def __init__(self, username):
		assert User.is_valid_username(username)
		self.username = username
		self.chr = Character(game.trait_pool)
		
	def is_valid_username(username):
		return username and len(username) > 0 and len(username) <= 16
	
	def is_ready(self):
		return self.chr.is_ready()

class Game:
	def __init__(self):
		self.trait_pool = create_trait_pool()
		self.users = {}
		self.max_users = chr_count
		self.fight_desc = ""

	def get_user(self, username):
		return self.users.get(username, None)

	def add_get_user(self, username):
		if User.is_valid_username(username):
			if username in self.users:
				return self.users[username]

			if len(self.users) < self.max_users:
				user = User(username)
				self.users[username] = user
				print(f"added user: username='{username}'")
				return user

		return None
	
	def get_ready_users(self):
		return [user for user in self.users.values() if user.is_ready()]
	
	def get_ready_users_count(self):
		return len(self.get_ready_users())

	def fight(self):
		self.fight_desc = gen_fight_desc([user.chr for user in self.users.values()])

	def get_response_creation(self, user):
		trait_offers = {
			trait_type.name: {
				"name": trait_type.name,
				"plural_name": trait_type.plural_name,
				"pick_count": trait_type.pick_count,
				"traits": [trait.desc for trait in user.chr.offers[trait_type]]
			}
			for trait_type in trait_types
		}
		return jsonify({
			"status": "creation",
			"trait_offers": trait_offers
		})
	
	def get_response_arena(self, user):
		chr_desc = user.chr.desc if user else "You are a spectator."
		return jsonify({
			"status": "arena",
			"chr_desc": chr_desc,
			"ready_users": {user.username: {
				"chr_name": user.chr.name
			 } for user in self.get_ready_users()},
			"max": self.max_users,
			"fight_desc": self.fight_desc
		})


game = Game()

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

@app.route("/connect", methods=["POST"])
@cross_origin()
def connect():
	global game
	data = request.json
	username = data.get("username")
	
	user = game.get_user(username)
	print(f"connection: username='{username}'{' (existing)' if user else ''}")

	if username == "":
		# Spectator.
		return game.get_response_arena(None)

	if not User.is_valid_username(username):
		print(f"invalid username: username='{username}'")
		return jsonify({
			"status": "error",
			"error": "Invalid username. Usernames must be between 1 and 16 characters in length."
		})

	user = game.add_get_user(username)

	if user and not user.chr.is_submitted():
		return game.get_response_creation(user)
	else:
		return game.get_response_arena(user)

@app.route("/create-character", methods=["POST"])
@cross_origin()
def create_character():
	global game
	data = request.json

	username = data.get("username")
	user = game.get_user(username)
	if not user:
		return jsonify({
			"status": "error",
			"error": "User not found."
		})

	try:
		trait_picks = data.get("trait_picks")
		for trait_type in trait_types:
			picks = trait_picks[trait_type.name]
			assert len(picks) == trait_type.pick_count
			offer = user.chr.offers[trait_type]
			for pick in picks:
				assert pick >= 0 and pick < len(offer)
				user.chr.pick_trait(trait_type, offer[pick])
	except Exception as e:
		print("Character creation failed.", e)
		return jsonify({
			"status": "error",
			"error": "Received invalid character creation data."
		})

	user.chr.name = "Name"
	user.chr.desc = gen_chr_desc(user.chr)

	if game.get_ready_users_count() == game.max_users:
		game.fight()

	return game.get_response_arena(user)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)