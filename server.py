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
				return user

		return None
	
	def get_ready_users_count(self):
		return len([user for user in self.users if user.chr.desc]) > self.max_users

	def fight(self):
		self.fight_desc = gen_fight_desc([user.chr for user in self.users])

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
		ready = self.get_ready_users_count()
		chr_desc = user.chr.desc if user else "You are a spectator."
		return jsonify({
			"status": "arena",
			"chr_desc": chr_desc,
			"ready": ready,
			"max": self.max_users,
			"fight_desc": self.fight_desc
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

	if username == "":
		# Spectator.
		return game.get_response_arena(None)

	if not User.is_valid_username(username):
		return jsonify({
			"status": "invalid_username"
		})

	user = game.add_get_user(username)

	if user and not user.chr.is_ready():
		return game.get_response_creation(user)
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
		user.chr.desc = gen_chr_desc(user.chr)

		if game.get_ready_users_count() == game.max_users:
			game.fight()

	return game.get_response_arena(user)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)