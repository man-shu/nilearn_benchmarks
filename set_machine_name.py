import json
import os
import sys

# accept the new machine name as a command line argument
if len(sys.argv) != 2:
    print("Usage: python set_machine_name.py <new_machine_name>")
    sys.exit(1)

new_machine_name = sys.argv[1]

json_file_path = "~/.asv-machine.json"
loaded_json = json.load(open(os.path.expanduser(json_file_path)))

# copy the first key into a new key new_machine_name
loaded_json[new_machine_name] = loaded_json[list(loaded_json.keys())[0]]
# also update "machine" value in the new key
loaded_json[new_machine_name]["machine"] = new_machine_name

# remove the old key
del loaded_json[list(loaded_json.keys())[0]]

# write the modified json back to the file
with open(os.path.expanduser(json_file_path), "w") as json_file:
    json.dump(loaded_json, json_file, indent=4)
