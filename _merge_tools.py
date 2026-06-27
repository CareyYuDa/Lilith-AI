
import os, sys
sys.path.insert(0, r'D:\Lilith\Lilith')

# Read tools.py
with open(r'D:\Lilith\Lilith\lilith_bot\tools.py', 'r', encoding='utf-8') as f:
    tools = f.read()

# Read evo tools
with open(r'D:\Lilith\Lilith\lilith_bot\_evo_tools.py', 'r', encoding='utf-8') as f:
    evo = f.read()

# Find insertion point
marker = 'LANGCHAIN_TOOLS = ['
idx = tools.index(marker)

# The evo tools should go before the list, and tools need to be added to the list
# Insert evo code before LANGCHAIN_TOOLS
new_content = tools[:idx] + '\n' + evo + '\n' + tools[idx:]

# Add new tools to LANGCHAIN_TOOLS list
new_content = new_content.replace(
    'LANGCHAIN_TOOLS = [',
    'LANGCHAIN_TOOLS = [\n    read_self_code, list_evolvable_files, evolve_self, review_evolution, rollback_evolution,'
)

with open(r'D:\Lilith\Lilith\lilith_bot\tools.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('OK: tools.py updated, ' + str(len(new_content)) + ' chars')
