use super::Vertex;
use glium;
use glium::backend::glutin_backend::GlutinFacade;
use glium::index::PrimitiveType;
use glium::Surface;
use hypernsga::cppn::{Cppn, GeometricActivationFunction, G};
use hypernsga::distribute::DistributeInterval;

pub fn render_cppn(
    display: &GlutinFacade,
    target: &mut glium::Frame,
    genome: &G,
    program: &glium::Program,
    viewport: glium::Rect,
) {
    // Layout the CPPN
    let cppn = Cppn::new(genome.network());
    let layers = cppn.group_layers();
    let mut dy = DistributeInterval::new(layers.len(), -1.0, 1.0);

    let mut cppn_node_positions: Vec<_> = genome
        .network()
        .nodes()
        .iter()
        .map(|_node| Vertex {
            position: [0.0, 0.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],
        })
        .collect();

    let mut cppn_node_triangles = Vec::new();

    let mut line_vertices = Vec::new();

    for layer in layers {
        let y = dy.next().unwrap();
        let mut dx = DistributeInterval::new(layer.len(), -1.0, 1.0);
        for nodeidx in layer {
            let x = dx.next().unwrap() as f32;
            let y = -y as f32;
            cppn_node_positions[nodeidx].position[0] = x;
            cppn_node_positions[nodeidx].position[1] = y;

            let node = &genome.network().nodes()[nodeidx];
            let w = 0.03;
            let aspect = viewport.width as f32 / viewport.height as f32;
            let h = aspect * w;

            match node.node_type().activation_function {
                GeometricActivationFunction::Linear => {
                    cppn_node_triangles.push(Vertex {
                        position: [x - (w / 2.0), y, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y + h, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                }
                GeometricActivationFunction::BipolarSigmoid => {
                    cppn_node_triangles.push(Vertex {
                        position: [x - (w / 2.0), y, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y + h, 0.0],
                        color: [1.0, 0.0, 1.0, 1.0],
                    });
                }
                GeometricActivationFunction::BipolarGaussian => {
                    cppn_node_triangles.push(Vertex {
                        position: [x - (w / 2.0), y, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y, 0.0],
                        color: [0.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x, y + h, 0.0],
                        color: [1.0, 0.0, 1.0, 1.0],
                    });
                }
                GeometricActivationFunction::Sine => {
                    cppn_node_triangles.push(Vertex {
                        position: [x, y, 0.0],
                        color: [1.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x - (w / 2.0), y + h, 0.0],
                        color: [1.0, 0.0, 1.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y + h, 0.0],
                        color: [1.0, 0.0, 1.0, 1.0],
                    });
                }
                _ => {
                    cppn_node_triangles.push(Vertex {
                        position: [x - (w / 2.0), y, 0.0],
                        color: [1.0, 0.0, 0.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x - (w / 2.0), y + h, 0.0],
                        color: [1.0, 0.0, 0.0, 1.0],
                    });
                    cppn_node_triangles.push(Vertex {
                        position: [x + (w / 2.0), y, 0.0],
                        color: [1.0, 0.0, 0.0, 1.0],
                    });

                    // cppn_node_triangles.push(Vertex{position: [x-(w/2.0), y+h, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    // cppn_node_triangles.push(Vertex{position: [x+(w/2.0), y+h, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    // cppn_node_triangles.push(Vertex{position: [x+(w/2.0),   y, 0.0], color:   [0.0, 0.0, 1.0, 1.0]});
                    //
                }
            }
        }
    }

    let mut cppn_links = Vec::new();
    genome.network().each_link_ref(|link_ref| {
        let src = link_ref.link().source_node_index().index();
        let dst = link_ref.link().target_node_index().index();
        cppn_links.push(src as u32);
        cppn_links.push(dst as u32);

        let src_x = cppn_node_positions[src].position[0];
        let src_y = cppn_node_positions[src].position[1];
        let dst_x = cppn_node_positions[dst].position[0];
        let dst_y = cppn_node_positions[dst].position[1];

        let weight = link_ref.link().weight().0 as f32;
        // assert!(weight.abs() <= 1.0); // XXX
        let weight = weight.abs().min(1.0);
        let wa = (weight.abs() / 2.0) + 0.5;
        let color = if weight >= 0.0 {
            [0.0, 1.0, 0.0, wa]
        } else {
            [1.0, 0.0, 0.0, wa]
        };

        line_vertices.push(Vertex {
            position: [src_x, src_y, 0.0],
            color: color,
        });
        line_vertices.push(Vertex {
            position: [dst_x, dst_y, 0.0],
            color: color,
        });
    });

    let _vertex_buffer_cppn = glium::VertexBuffer::new(display, &cppn_node_positions).unwrap();
    let _cppn_index_buffer =
        glium::IndexBuffer::new(display, PrimitiveType::LinesList, &cppn_links).unwrap();

    let triangle_buffer = glium::VertexBuffer::new(display, &cppn_node_triangles).unwrap();

    let uniforms_cppn = uniform! {
        matrix: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ]
    };

    let draw_parameters2 = glium::draw_parameters::DrawParameters {
        line_width: Some(3.0),
        blend: glium::Blend::alpha_blending(),
        smooth: Some(glium::draw_parameters::Smooth::Nicest),
        viewport: Some(viewport),
        ..Default::default()
    };

    // target.draw(&vertex_buffer_cppn, &glium::index::NoIndices(PrimitiveType::Points), program, &uniforms_cppn, &draw_parameters2).unwrap();
    // target.draw(&vertex_buffer_cppn, &cppn_index_buffer, program, &uniforms_cppn, &draw_parameters2).unwrap();

    let lines_buffer = glium::VertexBuffer::new(display, &line_vertices).unwrap();
    target
        .draw(
            &lines_buffer,
            &glium::index::NoIndices(PrimitiveType::LinesList),
            program,
            &uniforms_cppn,
            &draw_parameters2,
        )
        .unwrap();
    target
        .draw(
            &triangle_buffer,
            &glium::index::NoIndices(PrimitiveType::TrianglesList),
            program,
            &uniforms_cppn,
            &draw_parameters2,
        )
        .unwrap();
}
