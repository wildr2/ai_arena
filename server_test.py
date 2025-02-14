from server_lib import *

use_dummy_trait_data = False
write_dummy_trait_data = False
dummy_trait_data_path = "dummy_data/dummy_trait_pool.pkl"
use_dummy_chr_data = False
write_dummy_chr_data = False
dummy_chr_data_path = "dummy_data/dummy_chr_list.pkl"

def cli_offer(chr, trait_type):
	trait_str = f"{trait_type.plural_name}" if trait_type.pick_count > 1 else f"{trait_type.name}"
	log_header(f"Choose {trait_type.pick_count} {trait_str.capitalize()}")
	offer = chr.offers[trait_type]
	for i in range(len(offer)):
		print(f"\t{i + 1}. {offer[i]}")

	for i in range(trait_type.pick_count):
		pick = -1
		while pick < 0:
			pick_str = input(f"Type the number of the {trait_str}:")
			try:
				pick = int(pick_str) - 1
			except:
				pick = -1
			if pick < 0 or pick >= len(offer):
				pick = -1
			if offer[pick] in chr.picks[trait_type]:
				pick = -1
		chr.pick_trait(trait_type, offer[pick])

def cli_create_chr(trait_pool):
	chr = Character(trait_pool)
	log_header("Character Creation")
	for trait_type in trait_pool.traits.keys():
		cli_offer(chr, trait_type)
	print("Deliberating...")
	chr.desc = gen_chr_desc(chr, "Character Description")
	log_header("Character Created!")
	print(chr.desc)
	log_header("")
	return chr

chrs = CharacterList()
if use_dummy_chr_data:
	chrs = CharacterList.deserialize(open(dummy_chr_data_path, "rb"))
else:
	if use_dummy_trait_data:
		trait_pool = TraitPool.deserialize(open(dummy_trait_data_path, "rb"))
		# trait_pool.log()
	else:
		trait_pool = TraitPool()

		if write_dummy_trait_data:
			trait_pool.serialize(open(dummy_trait_data_path, "wb"))

	for i in range(chr_count):
		chrs.list.append(cli_create_chr(trait_pool))
	
	if write_dummy_chr_data:
		chrs.serialize(open(dummy_chr_data_path, "wb"))

input("Start the fight?")
print("Preparing the arena...")
fight_desc = gen_fight_desc(chrs)
log_header("Fight!")
print(fight_desc)