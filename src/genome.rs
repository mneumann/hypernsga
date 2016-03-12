use acyclic_network::{Network, NodeType, NodeIndex};
use weight::Weight;
use rand::Rng;

/// Genome representing a feed-forward (acyclic) network with node type `NT`.
#[derive(Clone, Debug)]
pub struct Genome<NT: NodeType> {
    /// Represents the acyclic feed forward network, with `NT` as node type and `Weight` for the edge weights.
    /// Other than that, nodes and edges do not have any further associated information (`()`).
    network: Network<NT, Weight, ()>,
}

impl<NT: NodeType> Genome<NT> {
    pub fn new() -> Self {
        Genome { network: Network::new() }
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

    fn random_node<R>(&self,
                      protected_nodes: usize, 
                      tournament_k: usize,
                      rng: &mut R) -> Option<NodeIndex>
        where R: Rng
    {
        assert!(tournament_k > 0);

        let n = self.network.node_count();
        if protected_nodes >= n {
            return None;
        }

        let mut min_node = NodeIndex::new(rng.gen_range(protected_nodes, n));
        let mut min_degree = self.network.node(min_node).degree();
        for _ in 1..tournament_k {
            let node_idx = NodeIndex::new(rng.gen_range(protected_nodes, n));
            let degree = self.network.node(node_idx).degree();
            if  degree < min_degree {
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
            (link.source_node_index(), link.target_node_index(), link.weight())
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
        self.add_link(middle_node, orig_target_node, second_link_weight.unwrap_or(orig_weight));  

        return true;
    }

    /// Structural Mutation `DropNode`.
    ///
    /// Choose a random node and remove it. All in and outgoing links to and from that node
    /// are removed as well. Note that the first `protected_nodes` are not removed!
    ///
    /// We choose `tournament_k` random nodes and remove the one with the smallest degree.

    pub fn mutate_drop_node<R>(&mut self,
                               protected_nodes: usize, 
                               tournament_k: usize,
                               rng: &mut R) -> bool
        where R: Rng
    {
        match self.random_node(protected_nodes, tournament_k, rng) {
            None => {
                false
            }
            Some(node_idx) => {
                self.network.remove_node(node_idx);
                true
            }
        }
    }

    /// Structural Mutation `ModifyNode`.
    ///
    /// Choose a random node and change it's type to `new_node_type`.
    ///
    /// We choose from `tournament_k` nodes the one with the smallest degree.

    pub fn mutate_modify_node<R>(&mut self,
                               new_node_type: NT,
                               protected_nodes: usize, 
                               tournament_k: usize,
                               rng: &mut R) -> bool
        where R: Rng
    {
        match self.random_node(protected_nodes, tournament_k, rng) {
            None => {
                false
            }
            Some(node_idx) => {
                self.network.node_mut(node_idx).set_node_type(new_node_type);
                true
            }
        }
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
}
