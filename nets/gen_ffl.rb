N = Integer(ARGV.shift)

INPUTS = (0...N).to_a
HIDDEN = (N...(N*2)).to_a
OUTPUTS = ((N*2)...(N*3)).to_a

WINPUT = "input"
WOUTPUT = "output"
WHIDDEN = "hidden"
#WINPUT = "0.0"
#WHIDDEN = "0.0"
#WOUTPUT = "0.0"

puts "graph"
puts "["
puts "  directed 1"
for i in INPUTS
    puts "  node [id #{i} weight #{WINPUT}]"
end
for i in HIDDEN
    puts "  node [id #{i} weight #{WHIDDEN}]"
end
for i in OUTPUTS
    puts "  node [id #{i} weight #{WOUTPUT}]"
end

for i in 0...N do
    puts "  edge [source #{INPUTS[i]} target #{HIDDEN[i]}]"
    puts "  edge [source #{HIDDEN[i]} target #{OUTPUTS[i]}]"
    puts "  edge [source #{INPUTS[i]} target #{OUTPUTS[i]}]"
end

puts "]"
