- [x] Start with a fully connected, but minimal, topology.
- [x] Start with x-axis seed nodes.
- [ ] Choose a higher mutation rate for individuals which have a worse fitness than
      the avergage.
- [ ] Adapt the mutation rate according to the average behavioral distance
- [ ] Weight mutation: Change until the behavioral distance to the original individual changes by 
      some percentage.  
- [ ] Implement a Conrod GUI for experimenting with setting configuration options
      during the simulation run. 
- [ ] Embed a RNG into every Genome.
- [ ] Record statistics, like number of mutations.
- [ ] Experiment with several different graphs.
- [ ] Make probability of structural mutation dependent on the complexity
      (number of nodes, number of links) of the genome.
- [ ] Substrate: Different placement
- [ ] Make weight mutation probability dependent on the current generation
- [ ] Make structural mutation dependent on the average node degree.
      For example, if there is a low connectivity of nodes, adding a new node is
      not a good thing.
- [ ] Add symmetric links, which, when updated, also update their counterpart.
- [ ] Add a fourth objective: Mutation work, which describes how much mutation has happened
      since the beginning for that individual.
- [ ] When adding a link, use a fixed weight for the second link
