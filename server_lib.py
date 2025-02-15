import ollama
from openai import OpenAI
import time
import re
import random
import pickle
from dataclasses import dataclass
import string

# Create api_keys.py locally.
import api_keys

chr_count = 2
provider = "openai"
model_name = {
	"ollama": "llama3.2:3b-instruct-q4_K_M",
	# "openai": "cognitivecomputations/dolphin3.0-mistral-24b:free"
	# "openai": "deepseek/deepseek-r1-distill-llama-70b:free",
	"openai": "mistralai/mistral-small-24b-instruct-2501:free",
}[provider]
ollama_context_length = 1024

debug_log = False
debug_no_model = False
use_dummy_trait_data = False
dummy_trait_data_path = "dummy_data/dummy_trait_pool.pkl"

prompt_chr_desc = """{0}\n\nThe above describes a character that will be forced to fight in the arena. Assume nothing that isn't described above. In second person tense, give the character a name and provide a very short summary of them. Begin with 'You are <name>'."""
prompt_fight = """{0}The above {1} characters will now be forced to fight to the death in the arena. Do not assume they are skilled fighters or that there abilities will be useful, rely only on the above descriptions. In present tense, give a very short account of what happens and who survives. Assume the reader is not familiar with the characters."""

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
		prompt = """Give me a choice of {0} fantasy character abilities. Ability descriptions should be specific and brief (just a few words), for example "you can win any debate". There should be a range of abilities from powerful, for example "you can summon a bolt of lightning", to underwhelming or funny, for instance "you can do a forward roll". Don't use exactly the above examples."""
	),
	TraitType(
		"weakness",
		"weaknesses",
		gen_count = 5,
		offer_count = 2,
		pick_count = 0,
		prompt = """Give me a choice of {0} fantasy character weaknesses. Weakness descriptions should be specific and brief (just a few words), for instance "you are terrible at throwing". Weaknesses can range from crippling, for instance "you are blind", to underwhelming or funny, for instance "you are allergic to peanuts". Do not assume that the character can use magic, or has a sword, etc. Don't use exactly the above examples."""
	),
	TraitType(
		"item",
		"items",
		gen_count = 5,
		offer_count = 3,
		pick_count = 0,
		prompt = """Give me a choice of {0} equipment a fantasy character could take into battle. Item descriptions should be specific and brief (just a few words), for instance "magic boots that make you run faster". Descriptions should not contain numbers. Items can range from powerful, for instance "a flaming sword", to underwhelming or funny, for instance "a pointy stick". Don't use exactly the above examples."""
	)
]
trait_types = [tt for tt in trait_types if tt.pick_count > 0]

class Generator():
	@dataclass(kw_only=True)
	class Options():
		temperature: float = 1.0
		
	def __init__(self, model_name):
		self.model_name = model_name

	def generate(self, prompt, options: Options):
		pass

class OllamaGenerator(Generator):
	def __init__(self, model_name, context_length):
		super().__init__(model_name)
		self.context_length = context_length

	def generate(self, prompt, options: Generator.Options):
		start_time = time.time()
		response: ollama.ChatResponse = ollama.generate(
			model=self.model_name,
			prompt=prompt,
			options={"num_ctx": self.context_length, "temperature": options.temperature}
		)
		content = response.response
		elapsed = time.time() - start_time
		return content, elapsed

class OpenAIGenerator(Generator):
	def __init__(self, model_name, api_key):
		super().__init__(model_name)
		self.client = OpenAI(
			base_url="https://openrouter.ai/api/v1",
			api_key=api_key
		)

	def generate(self, prompt, options: Generator.Options):
		start_time = time.time()
		response = self.client.chat.completions.create(
			model=self.model_name,
			messages=[{"role": "user", "content": prompt}],
			temperature=options.temperature
		)
		content = response.choices[0].message.content
		elapsed = time.time() - start_time
		return content, elapsed

def create_generator():
	if provider == "ollama":
		return OllamaGenerator(model_name, ollama_context_length)
	elif provider == "openai":
		return OpenAIGenerator(model_name, api_keys.openai_api_key)
	else:
		raise ValueError(f"Unknown provider: {provider}")

generator = create_generator()

def log_header(header):
	print(header.center(60, "-"))

def gen_traits(trait_type, total_start_time):
	if debug_no_model:
		return [Trait(f"{trait_type.name} {i}") for i in range(trait_type.gen_count * chr_count)]
	
	traits = []
	for i in range(chr_count):
		formatted_prompt = trait_type.prompt.format(trait_type.gen_count)
		options = Generator.Options(
			temperature=0.87
		)
		content, elapsed = generator.generate(formatted_prompt, options)

		if debug_log:
			log_header(f"{trait_type.name.capitalize()} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
			print(content)
			log_header("")
		
		new_traits = []
		for line in content.split("\n"):
			text = re.sub(r"\d+\.\s*", "", line)
			if text != line:
				text = text.replace("*", "").strip()
				new_traits.append(Trait(text))

		if len(new_traits) != trait_type.gen_count:
			print(f"{len(new_traits)} traits != {trait_type.gen_count}")
			log_header(f"{trait_type.name.capitalize()} {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
			print(content)
			log_header("")
			raise ValueError("Failed to generate traits.")
			
		traits.extend(new_traits)

		total_elapsed = time.time() - total_start_time
		print(f"{time.strftime('%H:%M:%S', time.gmtime(total_elapsed))} generated {len(traits)}/{trait_type.gen_count * chr_count} {trait_type.plural_name}")

	return traits
	
def gen_chr_desc(chr):
	if debug_no_model:
		return "Name", "You are..."
	
	formatted_prompt = prompt_chr_desc.format(chr.get_chr_sheet())
	options = Generator.Options(
		temperature=0.85
	)
	content, elapsed = generator.generate(formatted_prompt, options)

	desc = content.replace("*", "").strip()
	name = desc.split()[2] # You are <name>
	name = name.translate(str.maketrans('', '', string.punctuation)) # Strip puncutation.

	if debug_log:
		log_header(f"Character Description {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
		print(content)
		log_header("Name")
		print(name)

	if not desc:
		raise ValueError("Failed to generate desc.")
	if not name:
		raise ValueError("Failed to generate name.")
	
	return name, desc

def gen_fight_desc(chrs):
	if debug_no_model:
		return "The crowd goes wild..."

	chr_descs = ""
	for i in range(len(chrs)):
		chr_descs += f"Character {i+1}:\n{chrs[i].desc}\n\n"
		# chr_descs += f"Character {i+1}:\n{chrs[i].get_chr_sheet()}\n\n" # missing chr name
	formatted_prompt = prompt_fight.format(chr_descs, len(chrs))

	options = Generator.Options(
		temperature=0.8
	)
	content, elapsed = generator.generate(formatted_prompt, options)

	desc = content.replace("*", "").strip()

	if debug_log:
		log_header(f"Fight {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
		print(content)
		log_header("")
	
	if not desc:
		raise ValueError("Failed to generate fight.")
	
	return desc

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
			try:
				self.traits[trait_type.name] = gen_traits(trait_type, start_time)
			except:
				raise

	def log(self):
		log_header("Trait Pool")
		for trait_type in trait_types:
			log_header(trait_type.name.capitalize())
			for trait in self.traits[trait_type.name]: print(f"{trait}")
		log_header("")

	def serialize(self, file):
		pickle.dump(self, file)

	def deserialize(file):
		pool = pickle.load(file)
		for trait_type in trait_types:
			assert trait_type.name in pool.traits, f"TraitPool file '{file.name}' missing trait '{trait_type.name}'"
			assert len(pool.traits[trait_type.name]) >= trait_type.offer_count * chr_count
		return pool

class Character:
	def __init__(self, trait_pool):
		self.offers = {
			trait_type.name: self._claim_offer(trait_pool.traits[trait_type.name], trait_type.offer_count)
			for trait_type in trait_types
		}
		self.picks = {trait_type.name: [] for trait_type in trait_types}
		self.name = ""
		self.desc = ""

	def is_submitted(self):
		return all(len(self.picks[tt.name]) == tt.pick_count for tt in trait_types)
	
	def is_ready(self):
		return self.name and self.desc

	def get_chr_sheet(self):
		sheet = ""
		for trait_type in trait_types:
			sheet += f"{trait_type.name.capitalize()}:\n"
			for i, pick in enumerate(self.picks[trait_type.name]):
				sheet += f"{i+1}. {pick}\n"
		return sheet.strip()

	def submit(self, trait_picks):
		if self.is_submitted():
			raise ValueError("Already submitted.")

		for trait_type in trait_types:
			assert len(self.picks[trait_type.name]) == 0, "Unexpected picks state."

			picks = trait_picks[trait_type.name]
			picks = list(dict.fromkeys(picks))
			if len(picks) != trait_type.pick_count:
				raise ValueError(f"Invalid trait picks. picks: {trait_picks[trait_type.name]} pick_count: {trait_type.pick_count}")

			offer = self.offers[trait_type.name]
			for pick in picks:
				if pick < 0 or pick >= len(offer):
					raise ValueError("Invalid trait picks.")
				
		for trait_type in trait_types:
			picks = trait_picks[trait_type.name]
			offer = self.offers[trait_type.name]
			for pick in picks:
				self._pick_trait(trait_type, offer[pick])
				
	def gen_desc(self):
		assert self.is_submitted() and not self.is_ready()
		try:
			self.name, self.desc = gen_chr_desc(self)
		except Exception as e:
			raise

	def _pick_trait(self, trait_type, trait):
		assert trait not in self.picks[trait_type.name]
		assert len(self.picks[trait_type.name]) < trait_type.pick_count
		self.picks[trait_type.name].append(trait)

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