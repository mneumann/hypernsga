// Calculate fitness based on GraphSimilarity

use ::cppn::{Expression, G};
use ::network_builder::NetworkBuilder;
use ::domain_graph::{Neuron, NeuronNetworkBuilder, GraphSimilarity};
use ::substrate::{SubstrateConfiguration, Position};
use ::fitness::{Fitness, DomainFitness};

pub fn fitness<P>(genome: &G,
              expression: &Expression,
              substrate_config: &SubstrateConfiguration<P, Neuron>,
              fitness_eval: &GraphSimilarity)
              -> Fitness
    where P: Position
{

    let mut network_builder = NeuronNetworkBuilder::new();
    let (behavior, connection_cost, sat) =
        expression.express(genome, &mut network_builder, substrate_config);

    // Evaluate domain specific fitness
    let domain_fitness = fitness_eval.fitness(network_builder.network());

    Fitness {
        domain_fitness: domain_fitness,
        behavioral_diversity: 0.0, // will be calculated in `population_metric`
        connection_cost: connection_cost,
        behavior: behavior,
        age_diversity: 0.0, // will be calculated in `population_metric`
        saturation: sat.sum(),
        complexity: genome.complexity(),
    }
}
