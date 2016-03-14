use acyclic_network::{Network, NodeType, NodeIndex};
use weight::{Weight, WeightRange, WeightPerturbanceMethod};
use prob::Prob;
use rand::{Rng, SeedableRng};
use pcg::PcgRng;

/// Genome representing a feed-forward (acyclic) network with node type `NT`.
#[derive(Debug)]
pub struct Genome<NT: NodeType> {
    /// Represents the acyclic feed forward network, with `NT` as node type and `Weight` for the edge weights.
    /// Other than that, nodes and edges do not have any further associated information (`()`).
    network: Network<NT, Weight, ()>,

    /// The genome can contain some nodes which cannot be modified or removed. These are the first
    /// `protected_nodes` in `network`.
    protected_nodes: usize,

    /// Embedded random number generator.
    rng: PcgRng,
}

impl<NT: NodeType> Clone for Genome<NT> {
    fn clone(&self) -> Self {
        Genome {
            network: self.network.clone(),
            protected_nodes: self.protected_nodes,
            rng: self.rng.clone()
        }
    }
}

impl<NT: NodeType> Genome<NT> {
    pub fn new(rng_seed: [u64;2]) -> Self {
        Genome {
            network: Network::new(),
            protected_nodes: 0,
            rng: SeedableRng::from_seed(rng_seed),
        }
    }

    pub fn rng(&mut self) -> &mut PcgRng {
        &mut self.rng
    }

    pub fn protect_nodes(&mut self) {
        self.protected_nodes = self.network.node_count();
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

    fn random_node(&mut self, tournament_k: usize) -> Option<NodeIndex>
    {
        assert!(tournament_k > 0);
        let rng = &mut self.rng;

        let n = self.network.node_count();
        if n <= self.protected_nodes {
            return None;
        }

        let mut min_node = NodeIndex::new(rng.gen_range(self.protected_nodes, n));
        let mut min_degree = self.network.node(min_node).degree();
        for _ in 1..tournament_k {
            let node_idx = NodeIndex::new(rng.gen_range(self.protected_nodes, n));
            let degree = self.network.node(node_idx).degree();
            if degree < min_degree {
                min_node = node_idx;
                min_degree = degree;
            }
        }

        Some(min_node)
    }

    /// Structural Mutation `AddNode`.
    ///
    /// Choose a random link and split it into two by inserting a new node in the middle.
    ///
    /// Note that the new `node_type` should allow incoming and outgoing links! Otherwise
    /// this panics!

    pub fn mutate_add_node(&mut self,
                           node_type: NT,
                           second_link_weight: Option<Weight>)
                           -> bool
    {
        let link_index = match self.network.random_link_index(&mut self.rng) {
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

    pub fn mutate_drop_node(&mut self, tournament_k: usize) -> bool
    {
        match self.random_node(tournament_k) {
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

    pub fn mutate_modify_node(&mut self,
                              new_node_type: NT,
                              tournament_k: usize)
                              -> bool
    {
        if let Some(node_idx) = self.random_node(tournament_k) {
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

    pub fn mutate_connect(&mut self, link_weight: Weight) -> bool
    {
        match self.network.find_random_unconnected_link_no_cycle(&mut self.rng) {
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

    /// Structural Mutation `Disconnect`.
    ///
    /// Mutate the genome by removing a random link.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_disconnect(&mut self) -> bool
    {
        let rng = &mut self.rng;
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

    pub fn mutate_weights(&mut self,
                          mutate_element_prob: Prob,
                          weight_perturbance: &WeightPerturbanceMethod,
                          link_weight_range: &WeightRange)
                          -> isize
    {
        let rng = &mut self.rng;
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

    pub fn crossover_weights(&mut self, _partner: &Self) -> isize
    {
        unimplemented!()
    }
}
