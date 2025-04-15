import random
from task.base_taskmodule import base_taskmodule
from tqdm import tqdm
from discopy.frobenius import Diagram as FrobeniusDiagram, Ty as FrobeniusTy, Box as FrobeniusBox, Id as FrobeniusId, Spider as FrobeniusSpider
from discopy.grammar.pregroup import Diagram as DiagramGrammar, Ty as GrammarTy, Box as GrammarBox, Functor


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

# Define boxes for different actions
walks_box = lambda actor, direction: FrobeniusBox(f"walks_{direction}", actor.obj, actor.obj)(actor)

turns_box = lambda actor, turn_type: FrobeniusBox(f"turn_{turn_type}", actor.obj, actor.obj)(actor)

turns_to_box = lambda follower, followed: FrobeniusBox("turns_to", 
                follower.obj @ followed.obj, 
                follower.obj @ followed.obj)(follower, followed)


waves_box = lambda wave1, wave2: FrobeniusBox("waves", 
                wave1.obj @ wave2.obj, 
                wave1.obj @ wave2.obj)(wave1, wave2)

question_box = lambda actor1, actor2: FrobeniusBox("question", 
                actor1.obj @ actor2.obj, 
                answer_type)(actor1, actor2)

answer_type = FrobeniusTy("answer")

# Define proper conversion functions
def FrobeniusToGrammarFunctorObjects(ty):
    """Convert Frobenius Ty to Pregroup Ty"""
    result = GrammarTy()
    for t in ty:
        if str(t) in names:
            result @= GrammarTy("Actor")
        elif str(t) == "answer":
            result @= GrammarTy("bool")
        else:
            raise NotImplementedError(f"Type {t} not supported")
    return result
def FrobeniusToGrammarFunctorArrows(box):
    """Convert Frobenius Box to Pregroup Box"""
    dom = FrobeniusToGrammarFunctorObjects(box.dom)
    cod = FrobeniusToGrammarFunctorObjects(box.cod)
    return GrammarBox(box.name, dom, cod)

# Create the functor
FrobeniusToGrammarFunctor = Functor(FrobeniusToGrammarFunctorObjects, FrobeniusToGrammarFunctorArrows)

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

    def turns_to(self, actor):
        self.direction = actor.direction

    def turns(self, turn_direction):
        self.direction = self.turn_dict[turn_direction][self.direction]

class Story:
    def __init__(self, actors, n_sentences, n_directions=2):
        self.n_sentences = n_sentences
        self.events = ["turns_to", "turns", "waves"]
        self.turn_direction_choices = ["around"] if n_directions==2 else ["left", "right", "around"]
        self.story = []
        self.n_directions = n_directions
        self.question = []
        self.active_actors = []
        for actor in actors:
            self.init_actor(actor)
        self.diagram = None

    def init_actor(self, actor):
        self.active_actors.append(actor)
        self.story.append(("walks", actor, actor.start_direction))

    def event(self):
        ev = random.choices(self.events)[0]

        if ev == "turns_to":
            act1, act2 = random.sample(self.active_actors, 2)
            if self.story[-1] != ("turns_to", act1.name, act2.name):
                act1.turns_to(act2)
                self.story.append(("turns_to", act1.name, act2.name))
            else:
                self.event()

        elif ev == "turns":
            act = random.choice(self.active_actors)
            turn_direction = random.choice(self.turn_direction_choices)
            act.turns(turn_direction)
            self.story.append(("turns", act.name, turn_direction))
        
        elif ev == "waves":
            act1, act2 = random.sample(self.active_actors, 2)
            self.story.append(("waves", act1.name, act2.name))
        else:
            raise NotImplementedError()
    
    def build_diagram(self):
        # Create the types
        dom = FrobeniusTy()
        for actor in self.active_actors:
            dom @= FrobeniusTy(actor.name)
        codom = answer_type

        #Generate the Frobenius Diagram from the Story
        @FrobeniusDiagram.from_callable(dom, codom)
        def diagram(*args):
            actors = list(args)
            str_to_idx = {actor.name: i for i, actor in enumerate(self.active_actors)}

            for i in range(len(actors)):
                actors[i] = walks_box(actors[i], self.active_actors[i].start_direction)
            for event in self.story:
                if event[0] == "turns_to":
                    actors[str_to_idx[event[1]]], actors[str_to_idx[event[2]]] = turns_to_box(actors[str_to_idx[event[1]]], actors[str_to_idx[event[2]]])
                elif event[0] == "waves":
                    actors[str_to_idx[event[1]]], actors[str_to_idx[event[2]]] = waves_box(actors[str_to_idx[event[1]]], actors[str_to_idx[event[2]]])
                elif event[0] == "turns":
                    actors[str_to_idx[event[1]]] = turns_box(actors[str_to_idx[event[1]]], event[2])
                elif event[0] == "walks":
                    pass
                else:
                    raise NotImplementedError(f"Event {event} not supported")
            answer = question_box(actors[str_to_idx[self.question[0].name]], actors[str_to_idx[self.question[1].name]])
            
            return answer
        
        # Apply the functor to convert the diagram to a proper grammar diagram
        grammar_diagram = FrobeniusToGrammarFunctor(diagram)
        return grammar_diagram
    
    def generate(self):
        while len(self.story) < self.n_sentences:
            self.event()
        self.question = random.sample(self.active_actors, 2)
        # print([act.name for act in self.question])
        # print(self.question[0].direction)
        # print(self.question[1].direction)
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
                # print(answer)
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
        self.max_actors = 5 #10
        self.min_sents = 5
        self.max_sents = 5
        self.n_samples = 25
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

        diagrams = []
        labels = []

        for dir1 in ["north", "south"]:
            for dir2 in ["north", "south"]:
                for turn_bool_1 in [True, False]:
                    for turn_bool_2 in [True, False]:
                        for turns_to_bool in [True, False]:
                            for wave_bool in [True, False]:
                                story = Story([Actor("Alice"), Actor("Bob")], 0, 2)
                                story.active_actors[0].start_direction = dir1
                                story.active_actors[1].start_direction = dir2
                                story.question = [story.active_actors[0], story.active_actors[1]]
                                if wave_bool:
                                        story.story.append(("waves", "Alice", "Bob"))
                                if turn_bool_1:
                                    story.story.append(("turns", "Alice", "around"))
                                if turn_bool_2:
                                    story.story.append(("turns", "Bob", "around"))
                                if turns_to_bool:
                                    story.story.append(("turns_to", "Alice", "Bob"))
                                if wave_bool:
                                        story.story.append(("waves", "Alice", "Bob"))
                            
                                diagrams.append(story.build_diagram())
                                labels.append([1, 0] if ((dir1 == dir2) ^ (turn_bool_1 ^ turn_bool_2)) or turns_to_bool else [0, 1])
        
        
        diagrams[0].draw()

        return diagrams, labels

    def get_gates_to_analyse(self):
        gates = ["turns_to", "waves"]
        res = []
        for gate in gates:
            dom = FrobeniusTy("Alice") @ FrobeniusTy("Bob")
            codom = dom

            #Generate Frobenius Diagram for the gate
            @FrobeniusDiagram.from_callable(dom, codom)
            def diagram(alice, bob):
                if gate == "turns_to":
                  alice, bob = turns_to_box(alice, bob)
                elif gate == "waves":
                  alice, bob = waves_box(alice, bob)
                return alice, bob
            
            # Apply the functor to convert the diagram to a proper grammar diagram
            grammar_diagram = FrobeniusToGrammarFunctor(diagram)
            res.append((grammar_diagram, gate))

        return res