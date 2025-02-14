import ollama
from ollama import ChatResponse
import time
import re
from pprint import pprint, pformat
import random
import pickle

chr_count = 2
ollama_model_name = "llama3.2:3b-instruct-q4_K_M"
# ollama_model_name = "llama3.1:8b-instruct-q4_K_M"
model_context_length = 1024
debug_log = False
debug_no_model = True
use_dummy_trait_data = True
dummy_trait_data_path = "dummy_data/dummy_trait_pool.pkl"

prompt_chr_desc = """{0}\n\nThe above describes a character that will be forced to fight in the arena, despite having limited if any fighting experience. In second person tense, give the character a name (not too pretentious) and provide a concise summary of their abilities, weaknesses, and equipment."""

prompt_fight = """{0}The above {1} characters will now be forced to fight to the death in the arena until only one is left alive. Give a concise blow-by-blow account of what happens and who survives. Assume the reader is not familiar with the characters. Use present tense, and refer to characters in third person."""
# prompt_fight = """{0}Suppose the above {1} characters were forced to fight to the death in the arena. Who would survive? Show your work."""

class TraitType:
	def __init__(self, name, plural_name, gen_count, offer_count, pick_count, prompt):
		self.name = name
		self.plural_name = plural_name
		self.gen_count = gen_count
		self.offer_count = offer_count
		self.pick_count = pick_count
		self.prompt = prompt

trait_types = [
	TraitType(
		name = "ability",
		plural_name = "abilities",
		gen_count = 5,
		offer_count = 3,
		pick_count = 1,
		prompt = """Give me a choice of {0} fantasy character abilities. Ability descriptions should be specific and brief (just a few words), for example "you can win any debate". There should be a range of abilities from powerful, for example "you can summon a bolt of lightning", to underwhelming and potentially funny, for instance "you can do a forward roll". Don't use exactly the above examples."""
	),
	TraitType(
		"weakness",
		"weaknesses",
		gen_count = 5,
		offer_count = 2,
		pick_count = 1,
		prompt = """Give me a choice of {0} fantasy character weaknesses. Weakness descriptions should be specific and brief (just a few words), for instance "you are terrible at throwing". Weaknesses can range from crippling, for instance "you are blind", to underwhelming and potentially funny, for instance "you are allergic to peanuts". Do not assume that the character can use magic, or has a sword, etc. Don't use exactly the above examples."""
	),
	TraitType(
		"item",
		"items",
		gen_count = 5,
		offer_count = 3,
		pick_count = 1,
		prompt = """Give me a choice of {0} equipment a fantasy character could take into battle. Item descriptions should be specific and brief (just a few words), for instance "magic boots that make you run faster". Descriptions should not contain numbers. Items can range from powerful, for instance "a flaming sword", to underwhelming and potentially funny, for instance "a pointy stick". Don't use exactly the above examples."""
	)
]

def log_header(header):
	print(header.center(60, "-"))

def gen_traits(trait_type, total_start_time):
	if debug_no_model:
		return [Trait(f"{trait_type.name} {i}") for i in range(trait_type.gen_count * chr_count)]
	
	traits = []
	for i in range(chr_count):
		formatted_prompt = trait_type.prompt.format(trait_type.gen_count)
		start_time = time.time()
		response: ChatResponse = ollama.generate(
			model=ollama_model_name, 
			prompt=formatted_prompt, 
			options={"num_ctx": model_context_length, "repeat_penalty": 1.1, "temperature": 0.87})
		content = response.response
		elapsed = time.time() - start_time

		if debug_log:
			log_header(f"{trait_type.name.capitalize()} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
			print(content)
			log_header("")
		
		new_traits = []
		for line in content.split("\n"):
			text = re.sub(r"\d+\.\s*", "", line)
			if text != line:
				new_traits.append(Trait(text.strip()))

		if len(new_traits) != trait_type.gen_count:
			print(f"{len(new_traits)} traits != {trait_type.gen_count}")
			log_header(f"{trait_type.name.capitalize()} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
			print(content)
			log_header("")
			exit()
			
		traits.extend(new_traits)

		total_elapsed = time.time() - total_start_time
		print(f"{time.strftime('%H:%M:%S', time.gmtime(total_elapsed))} generated {len(traits)}/{trait_type.gen_count * chr_count} {trait_type.plural_name}")

	return traits
	
def gen_chr_desc(chr):
	if debug_no_model:
		return "You are..."
	
	formatted_prompt = prompt_chr_desc.format(chr.get_chr_sheet())
	start_time = time.time()
	response: ChatResponse = ollama.generate(
		model=ollama_model_name, 
		prompt=formatted_prompt, 
		options={"num_ctx": model_context_length, "repeat_penalty": 1.1, "temperature": 0.85})
	content = response.response
	elapsed = time.time() - start_time 

	if debug_log:
		log_header(f"Character Description {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
		print(content)
		log_header("")
	
	return content

def gen_fight_desc(chrs):
	if debug_no_model:
		return "The crowd goes wild..."

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

def create_trait_pool():
	if use_dummy_trait_data:
		return TraitPool.deserialize(open(dummy_trait_data_path, "rb"))
	else:
		return TraitPool()

class Trait:
	def __init__(self, desc):
		self.desc = desc
		self.chr = None

	def __str__(self):
		return self.desc

	def claim(self, chr):
		self.chr = chr

class TraitPool:
	def __init__(self):
		self.traits = {}
		start_time = time.time()
		for trait_type in trait_types:
			self.traits[trait_type] = gen_traits(trait_type, start_time)

	def log(self):
		log_header("Trait Pool")
		for trait_type in trait_types:
			log_header(trait_type.name.capitalize())
			for trait in self.traits[trait_type]: print(f"{trait}")
		log_header("")

	def serialize(self, file):
		pickle.dump(self, file)

	def deserialize(file):
		pool = pickle.load(file)
		for trait_type in trait_types:
			assert trait_type in pool.traits, f"TraitPool file '{file.name}' missing trait '{trait_type.name}'"
			assert len(pool.traits[trait_type]) >= trait_type.offer_count * chr_count
		return pool

	# Custom serialization
	def __getstate__(self):
		state = self.__dict__.copy()
		# Convert TraitType keys to names
		state["traits"] = {tt.name: traits for tt, traits in state["traits"].items()}
		return state

	# Custom dezerialization
	def __setstate__(self, state):
		# Convert names back to TraitType instances
		traits = state["traits"]
		self.traits = {}
		for name, trait_list in traits.items():
			# Find the corresponding TraitType instance
			tt = next((tt for tt in trait_types if tt.name == name), None)
			if tt is not None:
				self.traits[tt] = trait_list
			else:
				# No matching TraitType
				pass

class Character:
	def __init__(self, trait_pool):
		self.offers = {
			trait_type: self._claim_offer(trait_pool.traits[trait_type], trait_type.offer_count)
			for trait_type in trait_types
		}
		self.picks = {trait_type: [] for trait_type in trait_types}
		self.desc = ""

	def is_ready(self):
		return all(len(self.picks[tt]) == tt.pick_count for tt in trait_types)

	def get_chr_sheet(self):
		sheet = ""
		for trait_type in trait_types:
			sheet += f"{trait_type.name.capitalize()}:\n"
			for i, pick in enumerate(self.picks[trait_type]):
				sheet += f"{i+1}. {pick}\n"
		return sheet.strip()

	def pick_trait(self, trait_type, trait):
		assert trait not in self.picks[trait_type]
		assert len(self.picks[trait_type]) < trait_type.pick_count
		self.picks[trait_type].append(trait)

	def _claim_offer(self, traits, n):
		unclaimed_traits = [trait for trait in traits if not trait.chr]
		assert len(unclaimed_traits) >= n
		random.shuffle(unclaimed_traits)
		claim = unclaimed_traits[:n]
		for trait in claim:
			trait.claim(self)
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