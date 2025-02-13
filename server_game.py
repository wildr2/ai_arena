import ollama
from ollama import ChatResponse
import time
import re
from pprint import pprint, pformat
import random
import pickle

# ollama_model_name = "llama3.2:3b-instruct-q4_K_M"
ollama_model_name = "llama3.1:8b-instruct-q4_K_M"
model_context_length = 1024
use_dummy_option_data = False
write_dummy_option_data = False
dummy_option_data_path = "dummy_data/dummy_option_pool.pkl"
use_dummy_chr_data = False
write_dummy_chr_data = False
dummy_chr_data_path = "dummy_data/dummy_chr_list.pkl"
debug_log = False

chr_count = 2
# How many options to present.
ability_offer_count = 5
weakness_offer_count = 2
item_offer_count = 3
# How many options the player may choose.
ability_pick_count = 3
weakness_pick_count = 1
item_pick_count = 1
# How many options generate per character and add to the pool.
ability_gen_count = 5
weakness_gen_count = 5
item_gen_count = 5

prompt_ability = """Give me a choice of {0} fantasy character abilities. Ability descriptions should be specific and brief (just a few words), for example "you can win any debate". There should be a range of abilities from powerful, for example "you can summon a bolt of lightning", to underwhelming and potentially funny, for instance "you can do a forward roll". Don't use exactly the above examples."""

prompt_weakness = """Give me a choice of {0} fantasy character weaknesses. Weakness descriptions should be specific and brief (just a few words), for instance "you are terrible at throwing". Weaknesses can range from crippling, for instance "you are blind", to underwhelming and potentially funny, for instance "you are allergic to peanuts". Do not assume that the character can use magic, or has a sword, etc. Don't use exactly the above examples."""

prompt_item = """Give me a choice of {0} equipment a fantasy character could take into battle. Item descriptions should be specific and brief (just a few words), for instance "magic boots that make you run faster". Descriptions should not contain numbers. Items can range from powerful, for instance "a flaming sword", to underwhelming and potentially funny, for instance "a pointy stick". Don't use exactly the above examples."""

prompt_chr_desc = """{0}\n\nThe above describes a character that will be forced to fight in the arena, despite having limited if any fighting experience. In second person tense, give the character a name (not too pretentious) and provide a concise summary of their abilities, weaknesses, and equipment."""

prompt_fight = """{0}The above {1} characters will now be forced to fight to the death in the arena until only one is left alive. Give a concise blow-by-blow account of what happens and who survives. Assume the reader is not familiar with the characters. Use present tense, and refer to characters in third person."""
# prompt_fight = """{0}Suppose the above {1} characters were forced to fight to the death in the arena. Who would survive? Show your work."""

def log_header(header):
	print(header.center(60, "-"))
	
def gen_options(prompt, per_run_count, run_count, debug_log_header):
	options = []
	for i in range(run_count):
		formatted_prompt = prompt.format(per_run_count)
		start_time = time.time()
		response: ChatResponse = ollama.generate(
			model=ollama_model_name, 
			prompt=formatted_prompt, 
			options={"num_ctx": model_context_length, "repeat_penalty": 1.1, "temperature": 0.87})
		content = response.response
		elapsed = time.time() - start_time 

		if debug_log:
			log_header(f"{debug_log_header} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
			print(content)
			log_header("")
			
		new_options = []
		for line in content.split("\n"):
			text = re.sub(r"\d+\.\s*", "", line)
			if text != line:
				new_options.append(ChrOption(text.strip()))

		if len(new_options) != per_run_count:
			print(f"{len(new_options)} options != {per_run_count}")
			log_header(f"{debug_log_header} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
			print(content)
			log_header("")
			exit()
		options.extend(new_options)

	return options

def gen_chr_desc(chr, debug_log_header):
	formatted_prompt = prompt_chr_desc.format(chr.get_chr_sheet())
	start_time = time.time()
	response: ChatResponse = ollama.generate(
		model=ollama_model_name, 
		prompt=formatted_prompt, 
		options={"num_ctx": model_context_length, "repeat_penalty": 1.1, "temperature": 0.85})
	content = response.response
	elapsed = time.time() - start_time 

	if debug_log:
		log_header(f"{debug_log_header} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
		print(content)
		log_header("")
	
	return content

def gen_fight_desc(chrs):
	chr_descs = ""
	for i in range(len(chrs.list)):
		chr_descs += f"Character {i+1}:\n{chrs.list[i].desc}\n\n"
		# chr_descs += f"Character {i+1}:\n{chrs.list[i].get_chr_sheet()}\n\n" # missing chr name
	formatted_prompt = prompt_fight.format(chr_descs, len(chrs.list))
	start_time = time.time()
	response: ChatResponse = ollama.generate(
		model=ollama_model_name, 
		prompt=formatted_prompt, 
		options={"num_ctx": model_context_length, "repeat_penalty": 1.1, "temperature": 0.8})
	content = response.response
	elapsed = time.time() - start_time 

	if debug_log:
		log_header(f"Fight {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
		print(content)
		log_header("")
	
	return content

def cli_offer(chr, offer, picks, pick_count, type_str, type_plural_str, type_ind_article_str):
	obj_str = f"{type_plural_str}" if pick_count > 1 else f"{type_str}"
	log_header(f"Choose {pick_count} {obj_str.capitalize()}")
	for i in range(len(offer)):
		print(f"\t{i + 1}. {offer[i]}")

	for i in range(pick_count):
		pick = -1
		while pick < 0:
			pick_str = input(f"Type the number of {type_ind_article_str} {type_str}:")
			try:
				pick = int(pick_str) - 1
			except:
				pick = -1
			if pick < 0 or pick >= len(offer):
				pick = -1
			if offer[pick] in picks:
				pick = -1
		chr.pick_option(picks, offer[pick])

def cli_create_chr(option_pool):
	chr = Character(option_pool)
	log_header("Character Creation")
	cli_offer(chr, chr.ability_offer, chr.ability_picks, ability_pick_count, "ability", "abilities", "an")
	cli_offer(chr, chr.weakness_offer, chr.weakness_picks, weakness_pick_count, "weakness", "weaknesses", "a")
	cli_offer(chr, chr.item_offer, chr.item_picks, item_pick_count, "item", "items", "an")
	print("Deliberating...")
	chr.desc = gen_chr_desc(chr, "Character Description")
	log_header("Character Created!")
	print(chr.desc)
	log_header("")

	return chr

class ChrOption:
	def __init__(self, desc):
		self.desc = desc
		self.chr = None

	def __str__(self):
		return self.desc

	def claim(self, chr):
		self.chr = chr

class ChrOptionPool:
	def __init__(self):
		self.ability_options = gen_options(prompt_ability, ability_gen_count, chr_count, "Abilities")
		self.weakness_options = gen_options(prompt_weakness, weakness_gen_count, chr_count, "Weaknesses")
		self.item_options = gen_options(prompt_item, item_gen_count, chr_count, "Items")
		
	def log(self):
		log_header("Option Pool")
		log_header("Abilities")
		for option in self.ability_options: print(f"{option}")
		log_header("Weaknesses")
		for option in self.weakness_options: print(f"{option}")
		log_header("Items")
		for option in self.item_options: print(f"{option}")
		log_header("")

	def serialize(self, file):
		pickle.dump(self, file)

	def deserialize(file):
		return pickle.load(file)

class Character:
	def __init__(self, option_pool):
		# ChrOptions presented for this character.
		self.ability_offer = self._claim_offer(option_pool.ability_options, ability_offer_count)
		self.weakness_offer = self._claim_offer(option_pool.weakness_options, weakness_offer_count)
		self.item_offer = self._claim_offer(option_pool.item_options, item_offer_count)
		# ChrOptions the player chose out of the options.
		self.ability_picks = []
		self.weakness_picks = []
		self.item_picks = []
		self.desc = ""

	def get_chr_sheet(self):
		sheet = ""
		sheet += "Abilities:\n"
		for i in range(len(self.ability_picks)): sheet += f"{i+1}. {self.ability_picks[i]}\n"
		sheet += "Weaknesses:\n"
		for i in range(len(self.weakness_picks)): sheet += f"{i+1}. {self.weakness_picks[i]}\n"
		sheet += "Equipment:\n"
		for i in range(len(self.item_picks)): sheet += f"{i+1}. {self.item_picks[i]}\n"
		return sheet.strip()

	def pick_option(self, chr_picks, option):
		assert(option not in chr_picks)
		chr_picks.append(option)

	def _claim_offer(self, options, n):
		unclaimed_options = [option for option in options if not option.chr]
		assert(len(unclaimed_options) >= n)
		random.shuffle(unclaimed_options)
		claim = unclaimed_options[:n]
		for option in claim:
			option.claim(self)
		return claim

	def log(self):
		log_header("Character")
		print(self.get_chr_sheet())
		print(f"Desc:\n{self.desc}")
		log_header("")

class CharacterList:
	def __init__(self):
		self.list = []

	def serialize(self, file):
		pickle.dump(self, file)

	def deserialize(file):
		return pickle.load(file)

chrs = CharacterList()
if use_dummy_chr_data:
	chrs = CharacterList.deserialize(open(dummy_chr_data_path, "rb"))
else:
	if use_dummy_option_data:
		option_pool = ChrOptionPool.deserialize(open(dummy_option_data_path, "rb"))
		# option_pool.log()
	else:
		option_pool = ChrOptionPool()

		if write_dummy_option_data:
			option_pool.serialize(open(dummy_option_data_path, "wb"))

	for i in range(chr_count):
		chrs.list.append(cli_create_chr(option_pool))
	
	if write_dummy_chr_data:
		chrs.serialize(open(dummy_chr_data_path, "wb"))

input("Start the fight?")
print("Preparing the arena...")
fight_desc = gen_fight_desc(chrs)
log_header("Fight!")
print(fight_desc)