import copy
import difflib
import math
from typing import Any, List, Optional, TypedDict


class FormulationDict(TypedDict):
    """Typed dictionary describing the five-element optimization formulation schema."""
    type: Optional[Any]
    sets: List[Any]
    parameters: List[Any]
    variables: List[Any]
    objective: Optional[Any]
    constraints: List[Any]


class FormulationNode:
    """
    Node in the MCTS tree representing a partial or complete formulation
    """

    SETS_LEVEL = 1
    PARAMETERS_LEVEL = 2
    VARIABLES_LEVEL = 3
    OBJECTIVE_LEVEL = 4
    CONSTRAINTS_LEVEL = 5
    FINAL_LEVEL = CONSTRAINTS_LEVEL

    def __init__(
        self,
        parent: Optional["FormulationNode"] = None,
        level: int = 0,
        component: Any = None,
        max_child_num: int = 10,
    ):
        """Initialize the object and store the state required by later methods.
        
        Args:
            parent: Parent MCTS node for the node being initialized.
            level: Formulation layer index in the MCTS tree.
            component: Formulation component stored at this tree node.
            max_child_num: Maximum number of children allowed for this node.
        """
        self.parent = parent
        self.children: List["FormulationNode"] = []  # Maps component to child node
        self.visits = 0
        self.rewards = 0.0
        self.reward_m2 = 0.0
        self.mean_uncertainty = 0.0
        self.revision_count = 0
        self.level = level  # 0=root, 1=sets, 2=parameters, 3=variables, 4=objective, 5=constraints
        self.component = component  # Component added at this node

        # Store the complete formulation up to this node
        if parent is None:
            self.formulation: FormulationDict = {
                'type': None,
                'sets': [],
                'parameters': [],
                'variables': [],
                'objective': None,
                'constraints': []
            }
        else:
            self.formulation = copy.deepcopy(parent.formulation)
            self._add_component_to_formulation(component)

        # Track whether this node has been expanded
        self.expanded = False
        # Store generated components for this node
        self.available_components: List[Any] = []

        # Revision recommendation from LLM for next level (e.g., current level is 1, this is recommendation for level 2)
        self.need_revision = 0 
        self.uncertainty = 0 # lower is better
        # Maximum number of children
        self.max_child_num = max_child_num
        self.score_report = None
        self.score_components = None
        self.evidence = None
        self.composite_score = None
        self.review = None
        self.last_reward = None
        self.revision_requests = []
        if parent is not None:
            self.revision_requests = [
                copy.deepcopy(request)
                for request in getattr(parent, "revision_requests", [])
                if request.get("target_level", 0) > self.level
            ]
            if self.revision_requests:
                self.need_revision = 1

    def _add_component_to_formulation(self, component: Any):
        """Add the component to the appropriate section of the formulation"""
        if self.level == self.SETS_LEVEL:  # Sets level
            self.formulation['sets'].append(component)
        elif self.level == self.PARAMETERS_LEVEL:  # Parameters level
            self.formulation['parameters'].append(component)
        elif self.level == self.VARIABLES_LEVEL:  # Variables level
            self.formulation['variables'].append(component)
        elif self.level == self.OBJECTIVE_LEVEL:  # Objective level
            self.formulation['objective'] = component
        elif self.level == self.CONSTRAINTS_LEVEL:  # Constraints level
            self.formulation['constraints'].append(component)

    def is_fully_expanded(self):
        """Check if all possible children have been expanded"""
        # A node is fully expanded if :
        # needn't revision and it has been expanded and all generated components have been used
        if self.need_revision == 0:
            return self.expanded and len(self.children) == len(self.available_components)
        # need revision but the number of childs reach the maximum value
        elif self.need_revision == 1:
            return len(self.children) >= self.max_child_num
        else:
            return False

    def is_terminal(self):
        """Check if this is a terminal node (complete formulation)"""
        # A complete formulation has at least:
        # - One set
        # - One parameter
        # - One variable
        # - An objective function
        # - At least one constraint
        return (len(self.formulation['sets']) > 0 and
                len(self.formulation['parameters']) > 0 and
                len(self.formulation['variables']) > 0 and
                self.formulation['objective'] is not None and
                len(self.formulation['constraints']) > 0)

    def get_unexpanded_components(self):
        """Get components that haven't been used to create children yet using semantic similarity"""
        if not self.expanded:
            return []

        def is_semantically_similar(comp1, comp2, threshold=0.8):
            # Convert dictionaries to strings for comparison
            """Compare two candidate components using normalized textual similarity.
            
            Args:
                comp1: Input value used by this function.
                comp2: Input value used by this function.
                threshold: Input value used by this function.
            
            Returns:
                The value described in the function summary.
            """
            str1 = str(comp1)
            str2 = str(comp2)

            # Option 1: Using difflib's SequenceMatcher
            similarity = difflib.SequenceMatcher(None, str1, str2).ratio()

            return similarity >= threshold

        if not self.children:
            return self.available_components
        # Filter out components that are semantically similar to any child
        return [comp for comp in self.available_components
                if not any(is_semantically_similar(comp, child.component) for child in self.children)]

    def reward_variance(self):
        """Empirical variance of observed rewards at this node."""
        if self.visits <= 1:
            return 1.0
        return max(self.reward_m2 / (self.visits - 1), 1e-6)

    def get_selection_score(self, parent_visits, exploration_weight=2, uncertainty_weight=0.25):
        """Calculate a lightweight Bayes-UCB style score for this node."""
        if self.visits == 0:
            return float('inf')  # Encourage exploring unvisited nodes

        posterior_mean = self.rewards
        posterior_std = math.sqrt(
            self.reward_variance() / self.visits
            + uncertainty_weight * self.mean_uncertainty
            + 0.1 * self.revision_count / self.visits
        )
        exploration = exploration_weight * posterior_std * math.sqrt(math.log(max(parent_visits, 2)))

        return posterior_mean + exploration
