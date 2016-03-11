use innovation::{Innovation, InnovationRange};
use acyclic_network::{Network, NodeIndex, LinkRefItem};
pub use acyclic_network::NodeType;
use traits::{Distance, Genotype};
use weight::{Weight, WeightRange, WeightPerturbanceMethod};
use alignment_metric::AlignmentMetric;
use std::collections::BTreeMap;
use alignment::{Alignment, align_sorted_iterators, LeftOrRight};
use std::cmp;
use rand::Rng;
use crossover::ProbabilisticCrossover;
use std::convert::Into;
use std::ops::Range;
use prob::Prob;
use mutate::{MutateMethod, MutateMethodWeighting};
use std::marker::PhantomData;
use traits::Mate;

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AnyInnovation(usize);

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeInnovation(usize);

#[derive(Copy, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinkInnovation(usize);

impl Innovation for AnyInnovation {}

impl Into<NodeInnovation> for AnyInnovation {
    fn into(self) -> NodeInnovation {
        NodeInnovation(self.0)
    }
}

impl Into<LinkInnovation> for AnyInnovation {
    fn into(self) -> LinkInnovation {
        LinkInnovation(self.0)
    }
}

impl Innovation for NodeInnovation {}

impl Innovation for LinkInnovation {}

struct CombinedAlignmentMetric {
    node_metric: AlignmentMetric,
    link_metric: AlignmentMetric,
}

impl CombinedAlignmentMetric {
    fn new() -> Self {
        CombinedAlignmentMetric {
            node_metric: AlignmentMetric::new(),
            link_metric: AlignmentMetric::new(),
        }
    }
}

#[inline]
fn count_disjoint_or_excess<I: Innovation>(metric: &mut AlignmentMetric,
                                           range: &InnovationRange<I>,
                                           innovation: I) {
    if range.contains(&innovation) {
        metric.disjoint += 1;
    } else {
        metric.excess += 1;
    }
}

/// GlobalCache trait.
///
/// For example when creating a new link within a genome, we want
/// to check if that same link, defined by the node innovations of it's
/// source and target nodes, has already occured at some other place.

pub trait GlobalCache {
    fn get_or_create_link_innovation(&mut self,
                                     source_node: NodeInnovation,
                                     target_node: NodeInnovation)
                                     -> LinkInnovation;
    fn create_node_innovation(&mut self) -> NodeInnovation;
}

pub struct GlobalInnovationCache {
    node_innovation_counter: Range<usize>,
    link_innovation_counter: Range<usize>,
    // (src_node, target_node) -> link_innovation
    link_innovation_cache: BTreeMap<(NodeInnovation, NodeInnovation), LinkInnovation>,
}

impl GlobalInnovationCache {
    pub fn new() -> Self {
        GlobalInnovationCache {
            node_innovation_counter: Range {
                start: 0,
                end: usize::max_value(),
            },
            link_innovation_counter: Range {
                start: 0,
                end: usize::max_value(),
            },
            link_innovation_cache: BTreeMap::new(),
        }
    }
}

impl GlobalCache for GlobalInnovationCache {
    // XXX: Use #entry()
    fn get_or_create_link_innovation(&mut self,
                                     source_node: NodeInnovation,
                                     target_node: NodeInnovation)
                                     -> LinkInnovation {
        let key = (source_node, target_node);
        if let Some(&cached_innovation) = self.link_innovation_cache.get(&key) {
            return cached_innovation;
        }
        let new_innovation = LinkInnovation(self.link_innovation_counter.next().unwrap());
        self.link_innovation_cache.insert(key, new_innovation);
        new_innovation
    }

    fn create_node_innovation(&mut self) -> NodeInnovation {
        NodeInnovation(self.node_innovation_counter.next().unwrap())
    }
}


/// Genome representing a feed-forward (acyclic) network.
///
/// Each node is uniquely identified by it's Innovation number. Each link is sorted according it's
/// associated Innovation number.
///
/// We have to keep both the `network` and the `node_innovation_map` in sync. That is, whenever we
/// add or remove a node, we have to update both.
#[derive(Clone, Debug)]
pub struct Genome<NT: NodeType> {
    /// Represents the acyclic feed forward network.
    network: Network<NT, Weight, AnyInnovation>,

    /// Maps the external id (innovation number) which is globally allocated, to the internal
    /// network node index.
    node_innovation_map: BTreeMap<NodeInnovation, NodeIndex>,
}

impl<NT: NodeType> Genotype for Genome<NT> {}

impl<NT: NodeType> Genome<NT> {
    pub fn new() -> Self {
        Genome {
            network: Network::new(),
            node_innovation_map: BTreeMap::new(),
        }
    }

    pub fn visit_nodes<F>(&self, mut f: F)
        where F: FnMut(NodeInnovation, NT)
    {
        for node in self.network.nodes() {
            f(node.external_node_id().into(), node.node_type().clone());
        }
    }

    pub fn visit_active_links<F>(&self, mut f: F)
        where F: FnMut(NodeInnovation, NodeInnovation, Weight)
    {
        self.network.each_link_ref(|link_ref| {
            if link_ref.link().is_active() {
                f(link_ref.external_source_node_id().into(),
                  link_ref.external_target_node_id().into(),
                  link_ref.link().weight());
            }
        });
    }

    /// Counts the number of matching, disjoint and excess node innovation numbers between
    /// `left_genome` and `right_genome`.

    pub fn node_alignment_metric(left_genome: &Self, right_genome: &Self) -> AlignmentMetric {
        let mut node_metric = AlignmentMetric::new();
        node_metric.max_len = cmp::max(left_genome.node_innovation_map.len(),
                                       right_genome.node_innovation_map.len());

        let left = left_genome.node_innovation_map.keys();
        let right = right_genome.node_innovation_map.keys();

        align_sorted_iterators(left, right, Ord::cmp, |alignment| {
            match alignment {
                Alignment::Match(_l, _r) => {
                    node_metric.matching += 1;
                }
                Alignment::Disjoint(..) => {
                    node_metric.disjoint += 1;
                }
                Alignment::Excess(..) => {
                    node_metric.excess += 1;
                }
            }
        });

        node_metric
    }

    /// Aligns links of two genomes.

    fn align_links<F>(left_genome: &Self, right_genome: &Self, mut f: F)
        where F: FnMut(Alignment<LinkRefItem<NT, Weight, AnyInnovation>>)
    {
        let left_nodes = left_genome.node_innovation_map.iter();
        let right_nodes = right_genome.node_innovation_map.iter();

        let left_link_innov_range = left_genome.link_innovation_range();
        let right_link_innov_range = right_genome.link_innovation_range();

        let left_network = &left_genome.network;
        let right_network = &right_genome.network;

        align_sorted_iterators(left_nodes,
                               right_nodes,
                               |&(kl, _), &(kr, _)| Ord::cmp(kl, kr),
                               |node_alignment| {
                                   match node_alignment {
                                       Alignment::Match((_, &left_node_index),
                                                        (_, &right_node_index)) => {

                                           // Both nodes are topological identical. So the link innovations can
                                           // also match up.
                                           align_sorted_iterators(left_network.link_ref_iter_for_node(left_node_index),
                                               right_network.link_ref_iter_for_node(right_node_index),
                                               |left_link_ref, right_link_ref| 
                                               Ord::cmp(&left_link_ref.external_link_id(),
                                               &right_link_ref.external_link_id()),
                                               |link_alignment| {
                                                   match link_alignment { 
                                                       Alignment::Match(left_link_ref, right_link_ref) => {
                                                           f(Alignment::Match(left_link_ref, right_link_ref));
                                                       }
                                                       Alignment::Disjoint(link_ref, left_or_right) => {
                                                           f(Alignment::Disjoint(link_ref, left_or_right));
                                                       }
                                                       Alignment::Excess(left_link_ref, pos @ LeftOrRight::Left) => {
                                                           if right_link_innov_range.contains(&left_link_ref.external_link_id().into()) {
                                                               f(Alignment::Disjoint(left_link_ref, pos));
                                                           } else {
                                                               f(Alignment::Excess(left_link_ref, pos));
                                                           }
                                                       }
                                                       Alignment::Excess(right_link_ref, pos @ LeftOrRight::Right) => {
                                                           if left_link_innov_range.contains(&right_link_ref.external_link_id().into()) {
                                                               f(Alignment::Disjoint(right_link_ref, pos));
                                                           } else {
                                                               f(Alignment::Excess(right_link_ref, pos));
                                                           }
                                                       }
                                                   }
                                               });
                                       }

                                       // in general, if a node is disjoint (or excess), it's link innovations cannot match up!
                                       // XXX: Optimize: once we hit an excess link id, all remaining ids are excess as well.
                                       Alignment::Disjoint((_, &node_index), pos) |
                                       Alignment::Excess((_, &node_index), pos) => {
                                           let (net, range) = match pos {
                                               LeftOrRight::Left => {
                                                   (left_network, right_link_innov_range)
                                               }
                                               LeftOrRight::Right => {
                                                   (right_network, left_link_innov_range)
                                               }
                                           };

                                           for link_ref in net.link_ref_iter_for_node(node_index) {
                                               if range.contains(&link_ref.external_link_id()
                                                                          .into()) {
                                                   f(Alignment::Disjoint(link_ref, pos));
                                               } else {
                                                   f(Alignment::Excess(link_ref, pos));
                                               }
                                           }
                                       }
                                   }
                               });
    }

    /// Determine the genetic compatibility between `left_genome` and `right_genome` in terms of matching,
    /// disjoint and excess genes (both node and link genes), as well as weight distance.

    fn combined_alignment_metric(left_genome: &Self,
                                 right_genome: &Self)
                                 -> CombinedAlignmentMetric {
        let mut metric = CombinedAlignmentMetric::new();
        metric.node_metric.max_len = cmp::max(left_genome.network.node_count(),
                                              right_genome.network.node_count());
        metric.link_metric.max_len = cmp::max(left_genome.network.link_count(),
                                              right_genome.network.link_count());

        let left_nodes = left_genome.node_innovation_map.iter();
        let right_nodes = right_genome.node_innovation_map.iter();

        let left_link_innov_range = left_genome.link_innovation_range();
        let right_link_innov_range = right_genome.link_innovation_range();

        let left_network = &left_genome.network;
        let right_network = &right_genome.network;

        align_sorted_iterators(left_nodes,
                               right_nodes,
                               |&(kl, _), &(kr, _)| Ord::cmp(kl, kr),
                               |node_alignment| {
                                   match node_alignment {
                                       Alignment::Match((_, &left_node_index),
                                                        (_, &right_node_index)) => {
                                           metric.node_metric.matching += 1;

                                           // Both nodes are topological identical. So the link innovations can
                                           // also match up.
                                           align_sorted_iterators(left_network.link_iter_for_node(left_node_index),
                                               right_network.link_iter_for_node(right_node_index),
                                               |&(_, left_link), &(_, right_link)| 
                                               Ord::cmp(&left_link.external_link_id(),
                                               &right_link.external_link_id()),
                                               |link_alignment| {
                                                   match link_alignment { 
                                                       Alignment::Match((_, left_link), (_, right_link)) => {
                                                           // we have a link match!
                                                           metric.link_metric.matching += 1;

                                                           // add up the weight distance
                                                           metric.link_metric.weight_distance += (left_link.weight().0 - right_link.weight().0).abs();
                                                       }
                                                       Alignment::Disjoint(..) => {
                                                           // the link is locally disjoint (list of links of the node) 
                                                           metric.link_metric.disjoint += 1;
                                                       }
                                                       Alignment::Excess((_, left_link), LeftOrRight::Left) => {
                                                           count_disjoint_or_excess(&mut metric.link_metric, &right_link_innov_range, left_link.external_link_id().into());
                                                       }
                                                       Alignment::Excess((_, right_link), LeftOrRight::Right) => {
                                                           count_disjoint_or_excess(&mut metric.link_metric, &left_link_innov_range, right_link.external_link_id().into());
                                                       }
                                                   }
                                               });
                                       }

                                       // in general, if a node is disjoint (or excess), it's link innovations cannot match up!
                                       // XXX: Optimize: once we hit an excess link id, all remaining ids are excess as well.
                                       Alignment::Disjoint((_, &node_index), LeftOrRight::Left) => {
                                           metric.node_metric.disjoint += 1;

                                           for (_, link) in
                                               left_network.link_iter_for_node(node_index) {
                                               count_disjoint_or_excess(&mut metric.link_metric,
                                                                        &right_link_innov_range,
                                                                        link.external_link_id()
                                                                            .into());
                                           }
                                       }

                                       Alignment::Disjoint((_, &node_index), LeftOrRight::Right) => {
                                           {
                                               metric.node_metric.disjoint += 1;

                                               for (_, link) in
                                                   right_network.link_iter_for_node(node_index) {
                                                   count_disjoint_or_excess(&mut metric.link_metric, &left_link_innov_range, link.external_link_id().into());
                                               }
                                           }
                                       }

                                       Alignment::Excess((_, &node_index), LeftOrRight::Left) => {
                                           metric.node_metric.excess += 1;

                                           for (_, link) in
                                               left_network.link_iter_for_node(node_index) {
                                               count_disjoint_or_excess(&mut metric.link_metric,
                                                                        &right_link_innov_range,
                                                                        link.external_link_id()
                                                                            .into());
                                           }
                                       }

                                       Alignment::Excess((_, &node_index), LeftOrRight::Right) => {
                                           metric.node_metric.excess += 1;

                                           for (_, link) in
                                               right_network.link_iter_for_node(node_index) {
                                               count_disjoint_or_excess(&mut metric.link_metric,
                                                                        &left_link_innov_range,
                                                                        link.external_link_id()
                                                                            .into());
                                           }
                                       }
                                   }
                               });

        metric
    }

    /// Determine the genomes range of node innovations. If the genome
    /// contains no nodes, this will return `None`. Otherwise it will
    /// return the Some((min, max)).
    ///
    /// # Complexity
    ///
    /// This runs in O(log n).

    pub fn node_innovation_range(&self) -> InnovationRange<NodeInnovation> {
        let mut range = InnovationRange::empty();

        if let Some(&min) = self.node_innovation_map.keys().min() {
            range.insert(min);
        }
        if let Some(&max) = self.node_innovation_map.keys().max() {
            range.insert(max);
        }

        return range;
    }

    /// Determine the link innovation range for that Genome.
    ///
    /// # Complexity
    ///
    /// O(n) where `n` is the number of nodes.

    fn link_innovation_range(&self) -> InnovationRange<LinkInnovation> {
        let mut range = InnovationRange::empty();

        let network = &self.network;
        network.each_node_with_index(|_, node_idx| {
            if let Some(link) = network.first_link_of_node(node_idx) {
                range.insert(link.external_link_id());
            }
            if let Some(link) = network.last_link_of_node(node_idx) {
                range.insert(link.external_link_id());
            }
        });

        range.map(|i| LinkInnovation(i.0))
    }

    /// Returns a reference to the feed forward network.

    pub fn network(&self) -> &Network<NT, Weight, AnyInnovation> {
        &self.network
    }

    /// Add a link between `source_node` and `target_node`. Associates the new
    /// link with `link_innovation` and gives it `weight`.
    ///
    /// Does not check for cycles. Test for cycles before using this method!
    ///
    /// # Note
    ///
    /// Does not panic or abort if a link with the same link innovation is added.
    ///
    /// # Panics
    ///
    /// If one of `source_node` or `target_node` does not exist.
    ///
    /// If a link between these nodes already exists!
    ///
    /// # Complexity 
    ///
    /// This runs in O(k) + O(log n), where `k` is the number of edges of `source_node`.
    /// This is because we keep the edges sorted. `n` is the number of nodes, because
    /// we have to lookup the internal node indices from the node innovations.

    pub fn add_link(&mut self,
                    source_node: NodeInnovation,
                    target_node: NodeInnovation,
                    link_innovation: LinkInnovation,
                    weight: Weight) {
        self.add_link_with_active(source_node, target_node, link_innovation, weight, true);
    }

    pub fn add_link_with_active(&mut self,
                                source_node: NodeInnovation,
                                target_node: NodeInnovation,
                                link_innovation: LinkInnovation,
                                weight: Weight,
                                active: bool) {
        let source_node_index = self.node_innovation_map[&source_node];
        let target_node_index = self.node_innovation_map[&target_node];

        debug_assert!(!self.network.link_would_cycle(source_node_index, target_node_index));
        debug_assert!(self.network.valid_link(source_node_index, target_node_index).is_ok());

        let _link_index = self.network.add_link_with_active(source_node_index,
                                                            target_node_index,
                                                            weight,
                                                            AnyInnovation(link_innovation.0),
                                                            active);
    }

    /// Check if the link is valid and if it would construct a cycle.
    fn valid_link(&self, source_node: NodeInnovation, target_node: NodeInnovation) -> bool {
        let source_node_index = self.node_innovation_map[&source_node];
        let target_node_index = self.node_innovation_map[&target_node];

        self.network.valid_link(source_node_index, target_node_index).is_ok()
    }

    /// Check if the link is valid and if it would construct a cycle.
    fn valid_link_no_cycle(&self,
                           source_node: NodeInnovation,
                           target_node: NodeInnovation)
                           -> bool {
        let source_node_index = self.node_innovation_map[&source_node];
        let target_node_index = self.node_innovation_map[&target_node];

        self.network.valid_link(source_node_index, target_node_index).is_ok() &&
        (!self.network.link_would_cycle(source_node_index, target_node_index))
    }

    pub fn link_count(&self) -> usize {
        self.network.link_count()
    }

    fn _add_node(&mut self, node_innovation: NodeInnovation, node_type: NT) -> NodeIndex {
        if self.node_innovation_map.contains_key(&node_innovation) {
            panic!("Duplicate node_innovation");
        }

        let node_index = self.network.add_node(node_type, AnyInnovation(node_innovation.0));
        self.node_innovation_map.insert(node_innovation, node_index);
        return node_index;
    }

    /// Add a new node with external id `node_innovation` and of type `node_type`
    /// to the genome.
    ///
    /// # Panics
    ///
    /// Panics if a node with the same innovation already exists in the genome.

    pub fn add_node(&mut self, node_innovation: NodeInnovation, node_type: NT) {
        let _ = self._add_node(node_innovation, node_type);
    }

    pub fn has_node(&self, node_innovation: NodeInnovation) -> bool {
        self.node_innovation_map.contains_key(&node_innovation)
    }

    pub fn node_count(&self) -> usize {
        assert!(self.node_innovation_map.len() == self.network.node_count());
        return self.node_innovation_map.len();
    }

    /// Performs a crossover operation on the two genomes `left_genome` and `right_genome`,
    /// producing a new offspring genome.

    pub fn crossover<R: Rng>(left_genome: &Self,
                             right_genome: &Self,
                             c_nodes: &ProbabilisticCrossover,
                             c_links: &ProbabilisticCrossover,
                             rng: &mut R)
                             -> Self {
        let mut offspring = Genome::new();

        Genome::crossover_nodes(left_genome, right_genome, &mut offspring, c_nodes, rng);
        let (_, _) = Genome::crossover_links(left_genome, right_genome, &mut offspring, c_links, rng);

        return offspring;
    }

    /// Crossover the nodes of `left_genome` and `right_genome`. Either take a node from the
    /// left or the right, depending on randomness and `c`.

    fn crossover_nodes<R: Rng>(left_genome: &Self,
                               right_genome: &Self,
                               offspring: &mut Self,
                               c: &ProbabilisticCrossover,
                               rng: &mut R) {
        let left_nodes = left_genome.node_innovation_map.iter();
        let right_nodes = right_genome.node_innovation_map.iter();

        let left_network = &left_genome.network;
        let right_network = &right_genome.network;

        align_sorted_iterators(left_nodes,
                               right_nodes,
                               |&(kl, _), &(kr, _)| Ord::cmp(kl, kr),
                               |node_alignment| {
                                   match node_alignment {
                                       Alignment::Match((&ni_l, &left_node_index),
                                                        (&ni_r, &right_node_index)) => {
                                           // Both genomes have the same node gene (node innovation).
                                           // Either take the node type from the left genome or the right.

                                           debug_assert!(ni_l == ni_r);

                                           if c.prob_match_left.flip(rng) {
                                               // take from left
                                               offspring.add_node(ni_l, left_network.node(left_node_index).node_type().clone());
                                           } else {
                                               // take from right
                                               offspring.add_node(ni_r, right_network.node(right_node_index).node_type().clone());
                                           }
                                       }

                                       Alignment::Disjoint((&ni_l, &left_node_index),
                                                           LeftOrRight::Left) => {
                                           if c.prob_disjoint_left.flip(rng) {
                                               offspring.add_node(ni_l, left_network.node(left_node_index).node_type().clone());
                                           }
                                       }

                                       Alignment::Disjoint((&ni_r, &right_node_index),
                                                           LeftOrRight::Right) => {
                                           if c.prob_disjoint_right.flip(rng) {
                                               offspring.add_node(ni_r, right_network.node(right_node_index).node_type().clone());
                                           }
                                       }

                                       Alignment::Excess((&ni_l, &left_node_index),
                                                         LeftOrRight::Left) => {
                                           if c.prob_excess_left.flip(rng) {
                                               offspring.add_node(ni_l, left_network.node(left_node_index).node_type().clone());
                                           }
                                       }

                                       Alignment::Excess((&ni_r, &right_node_index),
                                                         LeftOrRight::Right) => {
                                           if c.prob_excess_right.flip(rng) {
                                               offspring.add_node(ni_r, right_network.node(right_node_index).node_type().clone());
                                           }
                                       }
                                   }
                               });
    }

    /// Crossover the links of `left_genome` and `right_genome`.

    fn crossover_links<R: Rng>(left_genome: &Self,
                               right_genome: &Self,
                               offspring: &mut Self,
                               c: &ProbabilisticCrossover,
                               rng: &mut R)
                               -> (usize, usize) {

        let mut total_nodes_added = 0;
        let mut total_links_added = 0;

        // First pass.
        //
        // Take all matching links from both genomes. We don't have to check for cycles
        // here, as each of the parents is acyclic and as such, the intersection of all links from
        // both genomes is acyclic as well.

        Genome::align_links(left_genome, right_genome, |link_alignment| {
            match link_alignment {
                Alignment::Match(left_link_ref, right_link_ref) => {
                    assert!(left_link_ref.external_link_id() == right_link_ref.external_link_id());
                    // A matching link that exists in both genomes.
                    // Either take it from left or right.
                    // Note that offspring already contains both the source and target node
                    // (assuming crossover_nodes() was called before) as also both of these
                    // nodes must exists in both parents.
                    let link_ref = if c.prob_match_left.flip(rng) {
                        // take link weight from left
                        left_link_ref
                    } else {
                        // take link weight from right
                        right_link_ref
                    };
                    // XXX: Modify weight

                    offspring.add_link_with_active(link_ref.external_source_node_id().into(),
                                                   link_ref.external_target_node_id().into(),
                                                   link_ref.external_link_id().into(),
                                                   link_ref.link().weight(),
                                                   link_ref.link().is_active());

                    total_links_added += 1;
                }
                _ => {
                    // NOTE: ignore all other links for now. In the second pass, we will take
                    // these into account.
                }
            }
        });


        // Second pass.
        //
        // Crossover disjoint and excess links. Two things to consider:
        //
        //    * Do not introduce cycles!
        //    * Nodes might not exist.

        Genome::align_links(left_genome, right_genome, |link_alignment| {
            // determine the probability
            let prob = match link_alignment {
                // Ignore matches. We already handled matching links in "Pass 1".
                Alignment::Match(_, _) => None,

                Alignment::Disjoint(_, LeftOrRight::Left) => Some(c.prob_disjoint_left),
                Alignment::Disjoint(_, LeftOrRight::Right) => Some(c.prob_disjoint_right),
                Alignment::Excess(_, LeftOrRight::Left) => Some(c.prob_excess_left),
                Alignment::Excess(_, LeftOrRight::Right) => Some(c.prob_excess_right),
            };

            if let Some(prob) = prob {
                match link_alignment {
                    Alignment::Match(_, _) => {
                        // Ignore. We already handled matching links in "Pass 1".
                    }

                    Alignment::Disjoint(link_ref, _) | Alignment::Excess(link_ref, _) => {
                        if prob.flip(rng) {
                            // Add nodes in case they do not exist.
                            // We take the nodeis from the genome the link belongs to.

                            let mut nodes_added = 0;
                            let source_id = link_ref.external_source_node_id().into();
                            let target_id = link_ref.external_target_node_id().into();

                            if !offspring.has_node(source_id) {
                                // add source node.
                                offspring.add_node(source_id,
                                                   link_ref.source_node().node_type().clone());
                                nodes_added += 1;
                            }

                            if !offspring.has_node(target_id) {
                                // add source node.
                                // We take the node from the genome the link belongs to.
                                offspring.add_node(target_id,
                                                   link_ref.target_node().node_type().clone());
                                nodes_added += 1;
                            }

                            debug_assert!(offspring.has_node(source_id) &&
                                          offspring.has_node(target_id));

                            let can_add_link = if nodes_added == 2 {
                                // Both nodes were added from the same genome as the link origins.
                                // This means that it is safe to add the link.
                                true
                            } else if nodes_added == 1 {
                                // Only one node was added from the genome.
                                // We have to check if the link is valid, for example a connection
                                // to an input node would be invalid. But we cannot introduce a cycle
                                // here.
                                // Actually, as all genomes share all input and output nodes, this
                                // should never be false!
                                offspring.valid_link(source_id, target_id)
                            } else {
                                // No node was added. We also have to check for cycles.
                                debug_assert!(nodes_added == 0);
                                offspring.valid_link_no_cycle(source_id, target_id)
                            };

                            total_nodes_added += nodes_added;

                            if can_add_link {
                                offspring.add_link_with_active(source_id,
                                                               target_id,
                                                               link_ref.external_link_id().into(),
                                                               link_ref.link().weight(),
                                                               link_ref.link().is_active());
                                total_links_added += 1;
                            }
                        }
                    }
                }

            }
        });

        return (total_nodes_added, total_links_added);
    }

    /// Mutate the genome by enabling a random disabled link.
    ///
    /// Note that we don't have to check for the introduction of cycles,
    /// as the cycle detection takes disabled links into account.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_enable_link<R: Rng>(&mut self, rng: &mut R) -> bool {
        match self.network.random_inactive_link_index(rng) {
            Some(idx) => {
                let ok = self.network.enable_link_index(idx);
                assert!(ok);
                true
            }
            None => false,
        }
    }

    /// Mutate the genome by adding a random link which is valid and does not introduce a cycle.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_add_link<R, G>(&mut self, link_weight: Weight, cache: &mut G, rng: &mut R) -> bool
        where R: Rng,
              G: GlobalCache
    {
        match self.network.find_random_unconnected_link_no_cycle(rng) {
            Some((source_node_idx, target_node_idx)) => {
                let ext_source_node_id: NodeInnovation = self.network
                                                             .node(source_node_idx)
                                                             .external_node_id()
                                                             .into();
                let ext_target_node_id: NodeInnovation = self.network
                                                             .node(target_node_idx)
                                                             .external_node_id()
                                                             .into();

                // Add new link to the offspring genome
                self.network.add_link(source_node_idx,
                                                target_node_idx,
                                                link_weight,
                                                AnyInnovation(cache.get_or_create_link_innovation(ext_source_node_id, ext_target_node_id).0));
                return true;
            }
            None => {
                return false;
            }
        }
    }

    /// Choose a random link. Split it in half creating a globally new node innovation!
    /// XXX: activate if link is inactive?
    /// XXX: random_active_link() is O(n) vs. O(1) for random_link().
    ///
    /// Note that the new `node_type` should allow incoming and outgoing links! Otherwise
    /// this panics!

    pub fn mutate_add_node<R, G>(&mut self,
                                 node_type: NT,
                                 second_link_weight: Weight,
                                 cache: &mut G,
                                 rng: &mut R)
                                 -> bool
        where R: Rng,
              G: GlobalCache
    {
        let link_index = match self.network.random_active_link_index(rng) {
            Some(idx) => idx,
            None => return false,
        };

        debug_assert!(self.network.link(link_index).is_active());

        // disable the original link gene (`link_index`).
        //
        // we keep this gene (but disable it), because this allows us to have a structurally
        // compatible genome to the original one, as disabled genes are taken into account for
        // the genomic distance measure.

        let _ok = self.network.disable_link_index(link_index);
        assert!(_ok);

        // Create new node innovation and add it to the genome.
        let new_node_innovation = cache.create_node_innovation();
        let new_node_index = self._add_node(new_node_innovation, node_type);

        // Add two new links connecting the three nodes. This cannot add a cycle!
        //
        // The first link reuses the same weight as the original link.
        // The second link uses `second_link_weight` as weight.
        // Ideally this is of full strenght. We want to make the modification
        // as little as possible.

        let orig_weight = self.network.link(link_index).weight();
        let source_node_index = self.network.link(link_index).source_node_index();
        let target_node_index = self.network.link(link_index).target_node_index();
        let source_node_innovation: NodeInnovation = self.network
                                                         .node(source_node_index)
                                                         .external_node_id()
                                                         .into();
        let target_node_innovation: NodeInnovation = self.network
                                                         .node(target_node_index)
                                                         .external_node_id()
                                                         .into();

        self.network
            .add_link(source_node_index,
                      new_node_index,
                      orig_weight,
                      AnyInnovation(cache.get_or_create_link_innovation(source_node_innovation,
                                                                        new_node_innovation)
                                         .0));

        self.network
            .add_link(new_node_index,
                      target_node_index,
                      second_link_weight,
                      AnyInnovation(cache.get_or_create_link_innovation(new_node_innovation,
                                                                        target_node_innovation)
                                         .0));
        return true;
    }


    /// Uniformly modify the weight of link genes, each with a probability of `mutate_prob`. It is
    /// guaranteed that this method makes a modification to at least one link (if it contains a
    /// link!).
    ///
    /// Returns the number of modifications
    ///
    /// Only modify enabled links

    pub fn mutate_link_weights_uniformly<R: Rng>(&mut self,
                                                 mutate_element_prob: Prob,
                                                 weight_perturbance: &WeightPerturbanceMethod,
                                                 link_weight_range: &WeightRange,
                                                 rng: &mut R)
                                                 -> usize {

        // Our network does not contain any links. Abort.
        if self.network.link_count() == 0 {
            return 0;
        }

        let mut modifications = 0;

        self.network.each_link_mut(|link| {
            if link.is_active() {
                if mutate_element_prob.flip(rng) {
                    let new_weight = weight_perturbance.perturb(link.weight(), link_weight_range, rng);
                    link.set_weight(new_weight);
                    modifications += 1;
                }
            }
        });

        if modifications == 0 {
            info!("no link modifications. use a random link");
            // Make at least one change to a randomly selected active link.
            let link_idx = self.network.random_active_link_index(rng).unwrap_or_else(|| self.network.random_link_index(rng).unwrap());
            let link = self.network.link_mut(link_idx);
            let new_weight = weight_perturbance.perturb(link.weight(), link_weight_range, rng);
            link.set_weight(new_weight);
            modifications += 1;
        }

        assert!(modifications > 0);
        return modifications;
    }

    /// Mutate the genome by removing a random link.
    ///
    /// Return `true` if the genome was modified. Otherwise `false`.

    pub fn mutate_delete_link<R: Rng>(&mut self, rng: &mut R) -> bool {
        match self.network.random_link_index(rng) {
            Some(idx) => {
                self.network.remove_link_at(idx);
                true
            }
            None => false,
        }
    }
}

/// This is used to weight a link AlignmentMetric.
#[derive(Debug)]
pub struct GenomeDistance {
    pub excess: f64,
    pub disjoint: f64,
    pub weight: f64,
}

impl<NT: NodeType> Distance<Genome<NT>> for GenomeDistance {
    fn distance(&self, genome_left: &Genome<NT>, genome_right: &Genome<NT>) -> f64 {
        let m = Genome::combined_alignment_metric(genome_left, genome_right).link_metric;

        if m.max_len == 0 {
            return 0.0;
        }

        self.excess * (m.excess as f64) / (m.max_len as f64) +
        self.disjoint * (m.disjoint as f64) / (m.max_len as f64) +
        self.weight *
        if m.matching > 0 {
            m.weight_distance / (m.matching as f64)
        } else {
            0.0
        }
    }
}

/// This trait is used to specialize link weight creation and node activation function creation.

pub trait ElementStrategy<NT: NodeType>
{
    fn link_weight_range(&self) -> WeightRange;
    fn full_link_weight(&self) -> Weight;
    fn random_node_type<R: Rng>(&self, rng: &mut R) -> NT;
}

/// Implementation for mating.

pub struct Mater<'a, N, S, C>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a,
          C: GlobalCache + 'a
{
    // probability for crossover. P_mutate = 1.0 - p_crossover
    pub p_crossover: Prob,
    pub p_crossover_detail_nodes: ProbabilisticCrossover,
    pub p_crossover_detail_links: ProbabilisticCrossover,
    pub p_mutate_element: Prob,
    pub weight_perturbance: WeightPerturbanceMethod,
    pub mutate_weights: MutateMethodWeighting,
    pub global_cache: &'a mut C,
    pub element_strategy: &'a S,
    pub _n: PhantomData<N>,
}

impl<'a, N, S, C> Mater<'a, N, S, C>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a,
          C: GlobalCache + 'a
{
    fn mutate<R: Rng>(&mut self,
                      offspring: &mut Genome<N>,
                      mutate_method: MutateMethod,
                      rng: &mut R)
                      -> bool {
        match mutate_method {
            MutateMethod::ModifyWeight => {
                let modifications =
                    offspring.mutate_link_weights_uniformly(self.p_mutate_element,
                                                            &self.weight_perturbance,
                                                            &self.element_strategy
                                                                 .link_weight_range(),
                                                            rng);

                modifications > 0
            }
            MutateMethod::AddConnection => {
                let link_weight = self.element_strategy
                                      .link_weight_range()
                                      .random_weight(rng);
                offspring.mutate_add_link(link_weight, self.global_cache, rng)
            }
            MutateMethod::EnableConnection => offspring.mutate_enable_link(rng),
            MutateMethod::DeleteConnection => offspring.mutate_delete_link(rng),
            MutateMethod::AddNode => {
                let second_link_weight = self.element_strategy.full_link_weight();
                let node_type = self.element_strategy.random_node_type(rng);
                offspring.mutate_add_node(node_type, second_link_weight, self.global_cache, rng)
            }
        }
    }
}


impl<'a, N, S, C> Mate<Genome<N>> for Mater<'a, N, S, C>
    where N: NodeType + 'a,
          S: ElementStrategy<N> + 'a,
          C: GlobalCache + 'a
{
    // Add an argument that descibes whether both genomes are of equal fitness.
    // Pass individual, which includes the fitness.
    fn mate<R: Rng>(&mut self,
                    parent_left: &Genome<N>,
                    parent_right: &Genome<N>,
                    prefer_mutate: bool,
                    rng: &mut R)
                    -> Genome<N> {
        if prefer_mutate == false && self.p_crossover.flip(rng) {
            Genome::crossover(parent_left, parent_right, &self.p_crossover_detail_nodes, &self.p_crossover_detail_links, rng)
        } else {
            // mutate
            let mutate_method = MutateMethod::random_with(&self.mutate_weights, rng);

            {
                let mut offspring = parent_left.clone();

                let modified = self.mutate(&mut offspring, mutate_method, rng);

                if modified {
                    return offspring;
                }
            }

            info!("no change in mutate left genome. mutate_method: {:?}", mutate_method);

            {
                let mut offspring = parent_right.clone();

                let modified = self.mutate(&mut offspring, mutate_method, rng);

                if modified {
                    return offspring;
                }

                info!("no change in mutate right genome");

                return offspring;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NodeType, Genome, NodeInnovation, LinkInnovation};
    use weight::Weight;
    use innovation::InnovationRange;

    #[derive(Clone, Debug)]
    struct NT;
    impl NodeType for NT {
        fn accept_incoming_links(&self) -> bool {
            true
        }
        fn accept_outgoing_links(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_add_node() {
        let mut genome = Genome::<NT>::new();
        assert_eq!(0, genome.node_count());
        genome.add_node(NodeInnovation(0), NT);
        assert_eq!(1, genome.node_count());
        genome.add_node(NodeInnovation(1), NT);
        assert_eq!(2, genome.node_count());
    }

    #[test]
    #[should_panic(expected = "Duplicate node_innovation")]
    fn test_add_duplicate_node() {
        let mut genome = Genome::<NT>::new();
        genome.add_node(NodeInnovation(0), NT);
        genome.add_node(NodeInnovation(0), NT);
    }

    #[test]
    fn test_add_link() {
        let mut genome = Genome::<NT>::new();
        let n0 = NodeInnovation(0);
        let n1 = NodeInnovation(1);
        let n2 = NodeInnovation(2);

        genome.add_node(n0, NT);
        genome.add_node(n1, NT);
        genome.add_node(n2, NT);

        assert_eq!(0, genome.link_count());

        genome.add_link(n0, n1, LinkInnovation(0), Weight(0.0));
        assert_eq!(1, genome.link_count());

        genome.add_link(n0, n2, LinkInnovation(0), Weight(0.0));
        assert_eq!(2, genome.link_count());
    }

    #[test]
    fn test_link_innovation_range() {
        let mut genome = Genome::<NT>::new();
        let n0 = NodeInnovation(0);
        let n1 = NodeInnovation(1);
        let n2 = NodeInnovation(2);

        genome.add_node(n0, NT);
        genome.add_node(n1, NT);
        genome.add_node(n2, NT);

        assert_eq!(InnovationRange::Empty, genome.link_innovation_range());

        genome.add_link(n0, n1, LinkInnovation(5), Weight(0.0));
        assert_eq!(InnovationRange::Single(LinkInnovation(5)),
                   genome.link_innovation_range());

        genome.add_link(n0, n2, LinkInnovation(1), Weight(0.0));
        assert_eq!(InnovationRange::FromTo(LinkInnovation(1), LinkInnovation(5)),
                   genome.link_innovation_range());

        genome.add_link(n1, n2, LinkInnovation(99), Weight(0.0));
        assert_eq!(InnovationRange::FromTo(LinkInnovation(1), LinkInnovation(99)),
                   genome.link_innovation_range());
    }

    #[test]
    fn test_node_innovation_range() {
        let mut genome = Genome::<NT>::new();
        assert_eq!(InnovationRange::Empty, genome.node_innovation_range());

        genome.add_node(NodeInnovation(5), NT);
        assert_eq!(InnovationRange::Single(NodeInnovation(5)),
                   genome.node_innovation_range());

        genome.add_node(NodeInnovation(7), NT);
        assert_eq!(InnovationRange::FromTo(NodeInnovation(5), NodeInnovation(7)),
                   genome.node_innovation_range());

        genome.add_node(NodeInnovation(6), NT);
        assert_eq!(InnovationRange::FromTo(NodeInnovation(5), NodeInnovation(7)),
                   genome.node_innovation_range());

        genome.add_node(NodeInnovation(4), NT);
        assert_eq!(InnovationRange::FromTo(NodeInnovation(4), NodeInnovation(7)),
                   genome.node_innovation_range());

        genome.add_node(NodeInnovation(1), NT);
        assert_eq!(InnovationRange::FromTo(NodeInnovation(1), NodeInnovation(7)),
                   genome.node_innovation_range());

        genome.add_node(NodeInnovation(1000), NT);
        assert_eq!(InnovationRange::FromTo(NodeInnovation(1), NodeInnovation(1000)),
                   genome.node_innovation_range());
    }

    #[test]
    fn test_node_align_metric() {
        let mut left = Genome::<NT>::new();
        let mut right = Genome::<NT>::new();

        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(0, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(0, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(5), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(1, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(10), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(2, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(2, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(6), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(2, m.max_len);
        assert_eq!(0, m.matching);
        assert_eq!(2, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(5), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(2, m.max_len);
        assert_eq!(1, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        left.add_node(NodeInnovation(6), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(3, m.max_len);
        assert_eq!(2, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(0, m.disjoint);
        assert_eq!(0.0, m.weight_distance);

        right.add_node(NodeInnovation(11), NT);
        let m = Genome::node_alignment_metric(&left, &right);
        assert_eq!(3, m.max_len);
        assert_eq!(2, m.matching);
        assert_eq!(1, m.excess);
        assert_eq!(1, m.disjoint);
        assert_eq!(0.0, m.weight_distance);
    }

    #[test]
    fn test_combined_align_metric() {
        let mut left = Genome::<NT>::new();
        let mut right = Genome::<NT>::new();

        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        left.add_node(NodeInnovation(5), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        left.add_node(NodeInnovation(10), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        right.add_node(NodeInnovation(6), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        right.add_node(NodeInnovation(5), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        left.add_node(NodeInnovation(6), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);

        right.add_node(NodeInnovation(11), NT);
        assert_eq!(Genome::node_alignment_metric(&left, &right),
                   Genome::combined_alignment_metric(&left, &right).node_metric);
    }

}
