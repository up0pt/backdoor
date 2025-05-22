WIP

sample command
```bash
uv run src/main.py \
--clients  30 \
--num_attackers  16 \
--attack_selection  pagerank \
--rounds  40 \
--pdr     0.7 \
--boost   5 \
--topology   barabasi \
--seed    123 \
--m       3 \
--lgclipping
```