INPUTS = Integer(ARGV.shift)
OUTPUTS = Integer(ARGV.shift)
puts "graph"
puts "["
puts "  directed 1"
  for i in 0...INPUTS
      puts "  node [id #{i} weight input]"
  end
  for i in INPUTS...(INPUTS+OUTPUTS)
      puts "  node [id #{i} weight output]"
  end
  for inp in 0...INPUTS 
      for outp in INPUTS...(INPUTS+OUTPUTS) 
        puts "  edge [source #{inp} target #{outp}]"
      end
  end
puts "]"
