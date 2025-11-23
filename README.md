# Ramanujan's Dreams

## Installation
For installation as a package run: `pip install git+https://github.com/UriKH/RamanujanDreams.git`  

## System general flow
The system running flow is similar to the following (WIP)

![CMF system flow (1)](https://github.com/user-attachments/assets/9d527043-e207-40e7-a9a6-2c59b3dc5b6d)

## Usage
Interaction with the system is via the System class (`from access import System`) and using the config files.
For easy usage preform accessing to configs, errors and functionality via `access.py`.

### Usage examples:
##### main.py
```
from rt_search import *

if __name__ =='__main__':
    config.configure(system={'LOGGING_BUFFER': 160}, analysis={'IDENTIFY_THRESHOLD': 0.1})
    analysis_config.PRINT_FOR_EVERY_SEARCHABLE = False

    results = System(
        dbs=[DBModV1(path='./my_database.db', json_path='./my_data.json')],
        analyzers=[AnalyzerModV1],
        searcher=SearcherModV1
    ).run(constants=['pi'])
```
**Note:** 
- The names of the constants should be writen as `sympy` calls them (`sp.pi`, `sp.E`, etc.).
- If you don't want to load or execute commadns using a JSON file, `json_path` could be ommited from the arguments.
- Changing configurations could be done in two ways:
  1. Using `config.configure(<config_section> = {<configuration-name> : <new value>})` - that way new configurations could be added to newly developed modules.
  2. Using each section's private configuration e.g. `db_config.USAGE = DBUsage.RETRIEVE_DATA`.

##### data.json
When reading this file, the system will execute the `append` command and will try to add the inspiration function ${}_2F_1(0.5)$ to set of inpiration funcitons for $\pi$ with the shift in start point as $x=0,~y=0,~z=\text{sp.Rational(1,2)}$.
```
{
    "command": "append",
    "data": [
        {
            "constant": "pi",
            "data": {
                "type": "pFq_formatter",
                "data": { "p": 2, "q": 1, "z": "1/2", "shifts": [0, 0, "1/2"] }
            }
        }
    ]
}
```

## Structure and Notes
### Structure:
The system is composed of 3 stages:
1. Database - storing and retrieving mapping from a constant to the inspiration functions.
2. Analysis - analysis of each of the CMFs i.e., filtering and prioritization of shards, borders, etc. 
3. Search - deep and full search within the searchable spaces. This stage (will) contain further logic and particularly ascend logic.

#### Configuration
Configuration management is done using distinct configuration files for each stage (or section - like `system`):
- `database` - `db_config`
- `analysis` - `analysis_config`
- `search` - `search_config`
- `system` - `sys_config`

Each `<X>_config` contains the configurations for this section. You can access those directly in order to view the current values.  
In order to change them you can use: `<X>_config.<property> = <new-value>`  
Or, by using the global configuration manager: `config.configure(<X> = {<property> : <new-value> })`  
The latter allows the **addition of new configurations**.


Globally the system looks as follows:
```
RT-CMF-sys
│   access.py
│
├─── system
├─── configs
├─── db_stage
├─── analysis_stage
├─── search_stage
└─── utils
```

### Database
| Stage                 | Path            | Description     | Example |
| :-------------:       | :-------------: | :-------------: | :-------: |
| Main folder           | `db_stage`      |  Contains a file with schemes of DB **module** and general DB + special errors file| |
| Modules               | `db_stage/DBs`  |  Each module directory contains an implemenation of the **module** in a `X_mod.py`, `config.py` and opionally an implementation of the service. | `db_v1` folder in which there are `db_mod.py`   (implementing the module DBMod class), `db.py` (implementing the DB)|
| Supported functions   |`db_stage/funcs` | All functions are used by the Formatter defined in `formatter.py`. Each inspiration function will be implemented in a file in the same folder. |            |
```
db_stage
│   db_scheme.py
│   errors.py
│
├─── DBs
│    └─── db_v1
│           config.py
│           db.py
│           db_mod.py
│
└─── funcs
        config.py
        formatter.py
        pFq_fmt.py
```

### Analysis
| Stage                 | Path            | Description     |
| :-------------:       | :-------------: | :-------------: |
| Main folder           | `analysis_stage`      | Contains a file with schemes of Analyzer **module** and general Analyzer + special errors file| 
| Modules               | `analysis_stage/analyzers`  |  See same row in _**Database**_ |
| Searchable spaces (e.g. shard)   |`analysis_stage/subspaces` | Each subspace has its own directory in which contains an implementaion of the subspace according to `searchable.py` + optional `config.py` | 
```
analysis_stage  
│    analysis_scheme.py  
│    errors.py  
│      
├─── analyzers  
│    └─── analyzer_v1  
│             analyzer.py  
│             analyzer_mod.py  
│             config.py  
│  
└─── subspaces  
     │   searchable.py
     │  
     └─── shard  
              shard.py  
              shard_extraction.py
 ```

### Searching
| Stage                 | Path            | Description     |
| :-------------:       | :-------------: | :-------------: |
| Main folder           | `search_stage`      | Contains a file with schemes of Searcher **module** and general Searcher | 
| Modules               | `search_stage/searchers`  |  See same row in _**Database**_ |
| Search methods    |`search_stage/methods` | Each method has its own directory in which contins an implementaion of the method according to `searcher_scheme.py` + optional `config.py` | 

```
search_stage
│   data_manager.py
│   searcher_scheme.py
│   errors.py
│
├─── methods
│    └─── serial
│             serial_searcher.py
│       
└─── searchers
     └─── searcher_v1
              config.py
              searcher_mod.py
```

### System and utils
- `system` contains system implementation, module implementation and system unique errors.
- `utils` contains utilities like logger and usfull type annotations for most of the code. `geometry` contains more specific utilities regarding geometrical calculations. 
```
system
    errors.py
    system.py
    module.py
utils
│   logger.py
│   types.py
│   
└─── geometry
        plane.py
        point_generator.py
        position.py
```



