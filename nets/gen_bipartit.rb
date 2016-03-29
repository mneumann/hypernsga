N = Integer(ARGV.shift)

INPUTS = (0...N).to_a
OUTPUTS = (N...(N*2)).to_a

WINPUT = "input"
WOUTPUT = "output"
#WINPUT = "0.0"
#WOUTPUT = "0.0"

puts "graph"
puts "["
puts "  directed 1"
for i in INPUTS
    puts "  node [id #{i} weight #{WINPUT}]"
end
for i in OUTPUTS
    puts "  node [id #{i} weight #{WOUTPUT}]"
end

INPUTS.each_with_index do |inp, idx| 
    puts "  edge [source #{inp} target #{OUTPUTS[idx]}]"
    puts "  edge [source #{inp} target #{OUTPUTS[idx-1]}]" if idx > 0 
end
puts "]"
