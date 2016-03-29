N = Integer(ARGV.shift)
winput = "input"
woutput = "output"
puts "graph"
puts "["
puts "  directed 1"
  for i in 0...N
      puts "  node [id #{i} weight #{winput}]"
  end
  for i in N...(2*N)
      puts "  node [id #{i} weight #{woutput}]"
  end

  for inp in 0...N  
      puts "  edge [source #{inp} target #{inp+N}]"
      if inp > 0
        puts "  edge [source #{inp} target #{inp+N-1}]"
      end
      if inp < N-1
        puts "  edge [source #{inp} target #{inp+N+1}]"
      end
  end
puts "]"
