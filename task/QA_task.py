import random
from task.base_taskmodule import base_taskmodule
from tqdm import tqdm
# from discopy.symmetric import Ty, Box
from discopy.frobenius import Diagram, Ty, Box, Id, Spider
from discopy.grammar.pregroup import Diagram as PreGroupDiagram, Ty as TyPreGroup, Box as BoxPreGroup, Functor 
# from discopy.grammar.pregroup import Diagram, Ty, Box, Id


names = [
    "Bob",
    "Alice",
    "Daniel",
    "Dorothy",
    "Paul",
    "Helen",
    "Jason",
    "Ruth",
    "Michael",
    "Linda",
    "Brian",
    "Donna",
    "Matthew",
    "Betty",
    "Charles",
    "Patricia",
    "James",
    "Susan",
    "George",
    "Sarah",
    "Richard",
    "Karen",
    "Christopher",
    "Nancy",
    "Steven",
    "Carol",
    "Kevin",
    "Anna",
    "Edward",
    "Lisa",
    "Eric",
    "Michelle",
    "Timothy",
    "Jennifer",
    "Robert",
    "Kimberly",
    "Mark",
    "Jessica",
    "David",
    "Laura",
    "Joseph",
    "Maria",
    "John",
    "Sharon",
    "William",
    "Elizabeth",
    "Andrew",
    "Emily",
    "Thomas",
    "Sandra",
    "Kenneth",
    "Mary",
    "Ben",
    "Margaret",
    "Jack",
    "Paula",
    "Ethan",
    "Natalie",
    "Peter",
    "Victoria",
    "Charlie",
]

actor_types = {name: Ty(name) for name in names}
actor_type = Ty("Actor")
actor_types_pregroup = {name: TyPreGroup(name) for name in names}

bool_type = Ty("bool")


class Actor:
    def __init__(self, name, direction=None, n_directions=2):
        self.name = name
        self.direction_choices = {
            2: ["north", "south"],
            4: ["north", "east", "south", "west"],
        }
        if direction is not None:
            self.direction = direction
        else:
            self.direction = random.choice(self.direction_choices[n_directions])
        self.start_direction = self.direction

        self.turn_dict = {
            "right": {
                "north": "east",
                "east": "south",
                "south": "west",
                "west": "north",
            },
            "around": {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            },
            "left": {
                "north": "west",
                "west": "south",
                "south": "east",
                "east": "north",
            },
        }

    def follows(self, actor):
        self.direction = actor.direction
        return f"{self.name} follows {actor.name}. "

    def opposite_direction_of(self, actor):
        self.direction = self.turn_dict["around"][actor.direction]
        return f"{self.name} goes in the opposite direction of {actor.name}. "

    def turns(self, turn_direction):
        self.direction = self.turn_dict[turn_direction][self.direction]
        return f"{self.name} turns {turn_direction}. "


class Story:
    def __init__(self, actors, n_sentences, n_directions=2):
        self.actor_init = actors
        self.n_actors = len(actors)
        self.n_sentences = n_sentences
        self.events = ["follows", "turns"] # "op_dir_of"
        self.turn_direction_choices = {
            2: ["around"],
            4: ["left", "right", "around"],
        }
        self.active_actors = []
        self.story = []
        self.n_directions = n_directions
        
        # Initialize diagram components
        self.actor_types = {}  # Maps actor name to its type
        self.diagram = None

    def init_actor(self, actor):
        if self.actor_init:
            self.active_actors.append(actor)
            self.story.append(f"{actor.name} walks {actor.direction}. ")
            # Add actor to diagram types
            self.actor_types[actor.name] = Ty(actor.name)

    def event(self):
        ev = random.choices(self.events)[0] # , weights=[p_2qb / 2, 1 - p_2q, p_2qb / 2]

        if ev == "follows":
            act1, act2 = random.sample(self.active_actors, 2)
            if self.story[-1] != f"{act1.name} follows {act2.name}. ":
                self.story.append((act1.follows(act2), ev, [self.active_actors.index(act1), self.active_actors.index(act2)]))
            else:
                self.event()

        elif ev == "turns":
            act = random.choice(self.active_actors)
            turn_direction = random.choice(self.turn_direction_choices[self.n_directions])
            self.story.append((act.turns(turn_direction), ev, [self.active_actors.index(act), turn_direction]))

        elif ev == "op_dir_of":
            act1, act2 = random.sample(self.active_actors, 2)
            if (
                self.story[-1]
                != f"{act1.name} goes in the opposite direction of {act2.name}. "
            ):  
                self.story.append((act1.opposite_direction_of(act2), ev, [self.active_actors.index(act1), self.active_actors.index(act2)]))
            else:
                self.event()
    
    def build_diagram(self):
        # Create the domain type - all actors
        dom = Ty()
        for actor in self.active_actors:
            dom @= actor_type #actor_types[actor.name]
        
        # Define boxes for different actions
        walks_box = lambda actor_name, direction: Box(f"walks_{direction}", actor_name.obj, actor_name.obj)

        turns_box = lambda actor_name, turn_type: Box(f"turn_{turn_type}", actor_name.obj, actor_name.obj)
        
        follows_box = lambda follower, followed: Box("follows", 
                      follower.obj @ followed.obj, 
                      follower.obj @ followed.obj)
        
        opposite_box = lambda actor1, actor2: Box("opposite_direction", 
                      actor1.obj @ actor2.obj, 
                      actor1.obj @ actor2.obj)
        
        forget_box = lambda actor_name: Box("forget", actor_name.obj, Ty())

        question_box = lambda actor1, actor2: Box("question", 
                      actor1.obj @ actor2.obj, 
                      Ty("bool"))
        
        @Diagram.from_callable(Ty(), bool_type)
        def diagram(*args):
            actors = [Spider(0,1,actor_type)() for actor in self.active_actors]
            # actors[0] = walks_box(actors[0], self.active_actors[0].start_direction)(actors[0])

            for i in range(len(actors)):
                actors[i] = walks_box(actors[i], self.active_actors[i].start_direction)(actors[i])
            for event in self.story:
                if event[1] == "follows":
                    actors[event[2][0]], actors[event[2][1]] = follows_box(actors[event[2][0]], actors[event[2][1]])(actors[event[2][0]], actors[event[2][1]])
                elif event[1] == "turns":
                    actors[event[2][0]] = turns_box(actors[event[2][0]], event[2][1])(actors[event[2][0]])
                elif event[1] == "op_dir_of":
                    actors[event[2][0]], actors[event[2][1]] = opposite_box(actors[event[2][0]], actors[event[2][1]])(actors[event[2][0]], actors[event[2][1]])  
            result = question_box(actors[self.active_actors.index(self.question[0])], actors[self.active_actors.index(self.question[1])])(actors[self.active_actors.index(self.question[0])], actors[self.active_actors.index(self.question[1])])
            
            return result
        
        # diagram.draw()
        
        # Define proper conversion functions
        def ob(ty):
            """Convert Frobenius Ty to Pregroup Ty"""
            if isinstance(ty, Ty):
                # Convert a compound type
                result = TyPreGroup()
                for t in ty:
                    # Convert each atomic type
                    if str(t) in actor_types_pregroup:
                        result @= actor_types_pregroup[str(t)]
                    elif str(t) == "bool":
                        result @= TyPreGroup("bool")
                    else:
                        result @= TyPreGroup(str(t))
                return result
            else:
                # Convert a single type
                return TyPreGroup(str(ty))
        
        def ar(box):
            """Convert Frobenius Box to Pregroup Box"""
            # Convert domain and codomain types
            dom = ob(box.dom)
            cod = ob(box.cod)
            # Create a new box with the same name but pregroup types
            return BoxPreGroup(box.name, dom, cod)
        
        # Create the functor
        F = Functor(ob, ar)
        
        # Apply the functor to convert the diagram
        grammar_diagram = F(diagram)
        
        # Draw and return the converted diagram
        # grammar_diagram.draw()
        # grammar_diagram = BoxPreGroup("question", TyPreGroup("bool"), TyPreGroup("bool"))
        return grammar_diagram
    
    def generate(self):
        for act in self.actor_init:
            self.init_actor(act)

        # while len(self.story) < self.n_sentences:
        #     self.event()
        
        self.question = random.sample(self.active_actors, 2)
        self.answer = self.question[0].direction == self.question[1].direction


        # Build the diagram representation of the story
        diagram = self.build_diagram()


        return self.story, diagram, self.answer

def gen_stories(
    min_actors, max_actors, min_sentences, max_sentences, n_samples,
    n_directions=2
):
    used_names = names[:max_actors]

    stories = []
    labels = []

    for n_act in tqdm(range(min_actors, max_actors + 1)):
        for n_sents in range(
            max(n_act, min_sentences), min(n_act * 6 + 1, max_sentences + 1)
        ):
            
            pos_count = 0
            neg_count = 0

            while (pos_count < n_samples or neg_count < n_samples):
                actors = [Actor(name=name, n_directions=n_directions) for name in used_names]
                story = Story(actors[:n_act], n_sents, n_directions=n_directions)
                s, diagram, answer = story.generate()
                # print(s)
                # diagram.draw()
                
                # Create label: [1,0] if same direction, [0,1] if different
                label = [1, 0] if answer else [0, 1]
                
                if answer and pos_count < n_samples:
                    stories.append(diagram)
                    labels.append(label)
                    pos_count += 1
                elif not answer and neg_count < n_samples:
                    stories.append(diagram)
                    labels.append(label)
                    neg_count += 1

    return stories, labels

class QA_task(base_taskmodule):
    """A task module for question-answering tasks"""

    def __init__(self):
        super().__init__()

        self.min_actors = 3
        self.max_actors = 3 #10
        self.min_sents = 5
        self.max_sents = 5
        self.n_samples = 50 #more samples for training necc
        self.n_directions = 2

        if self.max_sents < self.max_actors:
            raise Exception("max_sents should be >= max_actors")

    def get_scenarios(self):
        print("Generating stories...")
        diagrams, labels = gen_stories(
            self.min_actors, self.max_actors, self.min_sents, self.max_sents, self.n_samples,
            n_directions=self.n_directions
        )

        return diagrams, labels

    def get_hints(self):
        # Define boxes for different actions
        walks_box = lambda actor_name, direction: Box(f"walks_{direction}", actor_name.obj, actor_name.obj)

        turns_box = lambda actor_name, turn_type: Box(f"turn_{turn_type}", actor_name.obj, actor_name.obj)
        
        follows_box = lambda follower, followed: Box("follows", 
                      follower.obj @ followed.obj, 
                      follower.obj @ followed.obj)
        
        opposite_box = lambda actor1, actor2: Box("opposite_direction", 
                      actor1.obj @ actor2.obj, 
                      actor1.obj @ actor2.obj)
        
        forget_box = lambda actor_name: Box("forget", actor_name.obj, Ty())

        question_box = lambda actor1, actor2: Box("question", 
                      actor1.obj @ actor2.obj, 
                      Ty("bool"))
        diagrams = []
        labels = []

        for dir1 in ["north", "south"]:
            for dir2 in ["north", "south"]:
                for turn_bool_val in [True, False]:
                    for turn_bool_val2 in [True, False]:
                        for follower_bool_val in [True, False]:
                            @Diagram.from_callable(Ty(), bool_type)
                            def diagram(*args):
                                actors = [Spider(0,1,actor_type)(), Spider(0,1,actor_type)()]
                                actors[0] = walks_box(actors[0], dir1)(actors[0])
                                actors[1] = walks_box(actors[1], dir2)(actors[1])
                                if turn_bool_val:
                                    actors[0] = turns_box(actors[0], "around")(actors[0])
                                if turn_bool_val2:
                                    actors[1] = turns_box(actors[1], "around")(actors[1])
                                if follower_bool_val:
                                    actors[0], actors[1] = follows_box(actors[0], actors[1])(actors[0], actors[1])
                                result = question_box(actors[1], actors[0])(actors[1], actors[0])
                                return result
                        
                            diagrams.append(diagram)
                            labels.append([1, 0] if ((dir1 == dir2) ^ (turn_bool_val ^ turn_bool_val2)) or follower_bool_val else [0, 1])
        
        # Define proper conversion functions
        def ob(ty):
            """Convert Frobenius Ty to Pregroup Ty"""
            if isinstance(ty, Ty):
                # Convert a compound type
                result = TyPreGroup()
                for t in ty:
                    # Convert each atomic type
                    if str(t) in actor_types_pregroup:
                        result @= actor_types_pregroup[str(t)]
                    elif str(t) == "bool":
                        result @= TyPreGroup("bool")
                    else:
                        result @= TyPreGroup(str(t))
                return result
            else:
                # Convert a single type
                return TyPreGroup(str(ty))
        
        def ar(box):
            """Convert Frobenius Box to Pregroup Box"""
            # Convert domain and codomain types
            dom = ob(box.dom)
            cod = ob(box.cod)
            # Create a new box with the same name but pregroup types
            return BoxPreGroup(box.name, dom, cod)
        
        # Create the functor
        F = Functor(ob, ar)
        
        grammar_diagrams = []
        # Apply the functor to convert the diagram
        for diagram in diagrams:
            grammar_diagram = F(diagram)
            grammar_diagrams.append(grammar_diagram)
        grammar_diagrams[1].draw()

        return grammar_diagrams, labels

    def get_dictionary(self):
        return [("follows", 2), ("turn_around", 1), ("question", 2), ("walks_north", 1), ("walks_south", 1)]
    

    def get_type_strings(self):
        return ["bool"] + ["Actor"]
        

