use acyclic_network::{Network, NodeType, NodeIndex};
use weight::{Weight, WeightRange, WeightPerturbanceMethod};
use prob::Prob;
use rand::Rng;

/// Genome representing a feed-forward (acyclic) network with node type `NT`.
#[derive(Clone, Debug)]
pub struct Genome<NT: NodeType> {
    /// Represents the acyclic feed forward network, with `NT` as node type and `Weight` for the edge weights.
    /// Other than that, nodes and edges do not have any further associated information (`()`).
    network: Network<NT, Weight, ()>,

    /// The genome can contain some nodes which cannot be modified or removed. These are the first
    /// `protected_nodes` in `network`.
    protected_nodes: usize,

    /// Birth iteration 
    birth_iteration: usize,
}

impl<NT: NodeType> Genome<NT> {
    pub fn new(birth_iteration: usize) -> Self {
        Genome {
            network: Network::new(),
            protected_nodes: 0,
            birth_iteration: birth_iteration,
        }
    }

    pub fn fork(&self, birth_iteration: usize) -> Self {
        Genome {
            network: self.network.clone(),
            protected_nodes: self.protected_nodes,
            birth_iteration: birth_iteration,
        }
    }

    pub fn complexity(&self) -> f64 {
        self.network.node_count() as f64
    }

    pub fn age(&self, current_iteration: usize) -> usize {
        assert!(current_iteration >= self.birth_iteration);
        current_iteration - self.birth_iteration
    }

    pub fn birth_iteration(&self) -> usize {
        self.birth_iteration
    }

    pub fn protect_nodes(&mut self) {
        self.protected_nodes = self.network.node_count();
    }

    pub fn protected_nodes(&self) -> usize {
        self.protected_nodes
    }


    /// Returns a reference to the feed forward network.

    pub fn network(&self) -> &Network<NT, Weight, ()> {
        &self.network
    }

    /// Connects two previously unconnected nodes by adding a link between `source_node` and `target_node` using `weight`.
    ///
    /// Does not check for cycles. Test for cycles before using this method!
    ///
    /// # Panics
    ///
    /// If one of `source_node` or `target_node` does not exist.
    ///
    /// If a connection between these nodes already exists!
    ///
    /// # Complexity
    ///
    /// O(1) because we use `add_link_unorderd`.

    pub fn add_link(&mut self, source_node: NodeIndex, target_node: NodeIndex, weight: Weight) {
        debug_assert!(!self.network.link_would_cycle(source_node, target_node));
        debug_assert!(self.network.valid_link(source_node, target_node).is_ok());

        let _link_index = self.network.add_link_unordered(source_node, target_node, weight, ());
    }

    pub fn add_node(&mut self, node_type: NT) -> NodeIndex {
        self.network.add_node(node_type, ())
    }

    pub fn node_count(&self) -> usize {
        self.network.node_count()
    }

    fn random_mindeg_node<R>(&self,
                             tournament_k: usize,
                             start_from: usize,
                             rng: &mut R)
                             -> NodeIndex
        where R: Rng
    {
        assert!(tournament_k > 0);

        let n = self.network.node_count();
        let mut min_node = NodeIndex::new(rng.gen_range(start_from, n));
        let mut min_degree = self.network.node(min_node).degree();
        for _ in 1..tournament_k {
            let node_idx = NodeIndex::new(rng.gen_range(start_from, n));
            let degree = self.network.node(node_idx).degree();
            if degree < min_degree {
                min_node = node_idx;
                min_degree = degree;
            }
        }

        min_node
    }

    fn random_unprotected_node<R>(&self, tournament_k: usize, rng: &mut R) -> Option<NodeIndex>
        where R: Rng
    {
        assert!(tournament_k > 0);

        let n = self.network.node_count();
        if n <= self.protected_nodes {
            return None;
        }
        Some(self.random_mindeg_node(tournament_k, self.protected_nodes, rng))
    }

    /// Structural Mutation `AddNode`.
    ///
    /// Choose a random link and split it into two by inserting a new node in the middle.
    ///
    /// Note that the new `node_type` should allow incoming and outgoing links! Otherwise
    /// this panics!

    pub fn mutate_add_node<R>(&mut self,
                              node_type: NT,
                              second_link_weight: Option<Weight>,
                              rng: &mut R)
                              -> bool
        where R: Rng
    {
        let link_index = match self.network.random_link_index(rng) {
            Some(idx) => idx,
            None => return false,
        };

        let (orig_source_node, orig_target_node, orig_weight) = {
            let link = self.network.link(link_index);
            (link.source_node_index(),
             link.target_node_index(),
             link.weight())
        };

        // remove original link
        self.network.remove_link_at(link_index);

        // Add new "middle" node
        let middle_node = self.network.add_node(node_type, ());

        // Add two new links connecting the three nodes. This cannot add a cycle!
        //
        // The first link reuses the same weight as the original link.
        // The second link uses `second_link_weight` as weight (or if `None`, `orig_weight`).
        // Ideally this is of full strength as we want to make the modification
        // to the network (CPPN) as little as possible.

        self.add_link(orig_source_node, middle_node, orig_weight);
        self.add_link(middle_node,
                      orig_target_node,
                      second_link_weight.unwrap_or(orig_weight));

        return true;
    }

    /// Structural Mutation `DropNode`.
    ///
    /// Choose a random node and remove it. All in and outgoing links to and from that node
    /// are removed as well. Note that the first `protected_nodes` cannot be removed!
    ///
    /// We choose `tournament_k` random nodes and remove the one with the smallest degree.

    pub fn mutate_drop_node<R>(&mut self, tournament_k: usize, rng: &mut R) -> bool
        where R: Rng
    {
        match self.random_unprotected_node(tournament_k, rng) {
            None => false,
            Some(node_idx) => {
                self.network.remove_node(node_idx);
                true
            }
        }
    }

    /// Structural Mutation `ModifyNode`.
    ///
    /// Choose a random node and change it's type to `new_node_type`.
    /// Note that protected nodes are not modified.
    ///
    /// We choose from `tournament_k` nodes the one with the smallest degree.

    pub fn mutate_modify_node<R>(&mut self,
                                 new_node_type: NT,
                                 tournament_k: usize,
                                 rng: &mut R)
                                 -> bool
        where R: Rng
    {
        if let Some(node_idx) = self.random_unprotected_node(tournament_k, rng) {
            if self.network.node(node_idx).node_type() != &new_node_type {
                self.network.node_mut(node_idx).set_node_type(new_node_type);
                return true;
            }
        }
        return false;
    }

    /// Structural Mutation `Connect`.
    ///
    /// Mutate the genome by adding a random valid link which does not introduce a cycle.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_connect<R>(&mut self, link_weight: Weight, rng: &mut R) -> bool
        where R: Rng
    {
        match self.network.find_random_unconnected_link_no_cycle(rng) {
            Some((source_node, target_node)) => {
                // Add new link to the genome
                self.add_link(source_node, target_node, link_weight);
                return true;
            }
            None => {
                return false;
            }
        }
    }

    /// Structural Mutation `Symmetric Join`.
    ///
    /// Given a connection between A -> B, this will try to add
    /// a second connection A' -> B with negative weight of A -> B.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_symmetric_join<R>(&mut self, rng: &mut R) -> bool
        where R: Rng
    {
        let link_index = match self.network.random_link_index(rng) {
            Some(idx) => idx,
            None => return false,
        };

        let (source_node, target_node, weight) = {
            let link = self.network.link(link_index);
            (link.source_node_index(),
             link.target_node_index(),
             link.weight())
        };

        for i in 0..self.network.node_count() {
            let node_idx = NodeIndex::new(i);
            if node_idx != source_node && node_idx != target_node {
                if self.network.valid_link(node_idx, target_node).is_ok() &&
                   !self.network.has_link(node_idx, target_node) &&
                   !self.network.link_would_cycle(node_idx, target_node) {
                    self.add_link(node_idx, target_node, weight.inv());
                    return true;
                }
            }
        }

        return false;
    }

    /// Structural Mutation `Symmetric Fork`.
    ///
    /// Given a connection between A -> B, this will try to add
    /// a second connection A -> B' with negative weight of A -> B.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_symmetric_fork<R>(&mut self, rng: &mut R) -> bool
        where R: Rng
    {
        let link_index = match self.network.random_link_index(rng) {
            Some(idx) => idx,
            None => return false,
        };

        let (source_node, target_node, weight) = {
            let link = self.network.link(link_index);
            (link.source_node_index(),
             link.target_node_index(),
             link.weight())
        };

        for i in 0..self.network.node_count() {
            let node_idx = NodeIndex::new(i);
            if node_idx != source_node && node_idx != target_node {
                if self.network.valid_link(source_node, node_idx).is_ok() &&
                   !self.network.has_link(source_node, node_idx) &&
                   !self.network.link_would_cycle(source_node, node_idx) {
                    self.add_link(source_node, node_idx, weight.inv());
                    return true;
                }
            }
        }

        return false;
    }


    /// Structural Mutation `Symmetric Connect`.
    ///
    /// Mutate the genome by adding two random links to the same node with
    /// symmetric weights (-weight, +weight) which does not introduce a cycle.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_symmetric_connect<R>(&mut self,
                                       link_weight: Weight,
                                       retries: usize,
                                       rng: &mut R)
                                       -> bool
        where R: Rng
    {
        match self.network.find_random_unconnected_link_no_cycle(rng) {
            Some((node1, node2)) => {
                // node1 and node2 are neither directly connected, nor do they form a
                // cycle when connected.
                // find a third node node3, which is neither connected to node1 nor node2.

                for _ in 0..retries {
                    let node3 = self.random_mindeg_node(3, 0, rng);
                    if self.network.valid_link(node1, node3).is_ok() &&
                       self.network.valid_link(node2, node3).is_ok() &&
                       !self.network.has_link(node1, node3) &&
                       !self.network.has_link(node2, node3) &&
                       !self.network.link_would_cycle(node1, node3) &&
                       !self.network.link_would_cycle(node2, node3) {
                        self.add_link(node1, node3, link_weight);
                        self.add_link(node2, node3, link_weight.inv());
                        return true;
                    }
                }
            }
            _ => {}
        }
        return false;
    }


    /// Structural Mutation `Disconnect`.
    ///
    /// Mutate the genome by removing a random link.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_disconnect<R>(&mut self, rng: &mut R) -> bool
        where R: Rng
    {
        match self.network.random_link_index(rng) {
            Some(idx) => {
                self.network.remove_link_at(idx);
                true
            }
            None => false,
        }
    }


    /// Uniformly modify the weight of links, each with a probability of `mutate_element_prob`. It
    /// the genome contains at least one link, it is guaranteed that this method makes a modification.
    ///
    /// Returns the number of modifications (if negative, indicates that we used a random link).

    pub fn mutate_weights<R>(&mut self,
                             mutate_element_prob: Prob,
                             weight_perturbance: &WeightPerturbanceMethod,
                             link_weight_range: &WeightRange,
                             rng: &mut R)
                             -> isize
        where R: Rng
    {
        let mut modifications = 0;

        self.network.each_link_mut(|link| {
            if mutate_element_prob.flip(rng) {
                let new_weight = weight_perturbance.perturb(link.weight(), link_weight_range, rng);
                link.set_weight(new_weight);
                modifications += 1;
            }
        });

        if modifications == 0 {
            // Make at least one change to a randomly selected link.
            if let Some(link_idx) = self.network.random_link_index(rng) {
                let link = self.network.link_mut(link_idx);
                let new_weight = weight_perturbance.perturb(link.weight(), link_weight_range, rng);
                link.set_weight(new_weight);
                modifications -= 1;
            }
        }

        return modifications;
    }

    pub fn crossover_weights<R>(&mut self, _partner: &Self, _rng: &mut R) -> isize
        where R: Rng
    {
        unimplemented!()
    }
}
