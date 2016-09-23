use ::glium;
use glium::backend::glutin_backend::GlutinFacade;
use glium::index::PrimitiveType;
use glium::Surface;
use viz_network_builder::VizNetworkBuilder;
use hypernsga::cppn::{G, Expression};
use hypernsga::substrate::{SubstrateConfiguration, Position3d};
use hypernsga::domain_graph::Neuron;
use hypernsga::network_builder::NetworkBuilder;
use transformation::Transformation;

pub fn render_graph(display: &GlutinFacade,
                    target: &mut glium::Frame,
                    genome: &G,
                    expression: &Expression,
                    program: &glium::Program,
                    transformation: &Transformation,
                    substrate_config: &SubstrateConfiguration<Position3d, Neuron>,
                    viewport: glium::Rect,
                    line_width: f32,
                    point_size: f32) {
    let mut network_builder = VizNetworkBuilder::new();
    let (_, _, _) = expression.express(&genome, &mut network_builder, &substrate_config);

    let vertex_buffer = glium::VertexBuffer::new(display, &network_builder.point_list).unwrap();

    let line_index_buffer = glium::IndexBuffer::new(display,
                                                    PrimitiveType::LinesList,
                                                    &network_builder.link_index_list)
        .unwrap();

    let rx = transformation.rotate_x.to_radians();
    let ry = transformation.rotate_y.to_radians();
    let rz = transformation.rotate_z.to_radians();
    let sx = transformation.scale_x;
    let sy = transformation.scale_y;
    let sz = transformation.scale_z;

    let perspective = {
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0f32]]
    };

    let uniforms_substrate = uniform! {
        matrix: [
            [sx*ry.cos()*rz.cos(), -ry.cos()*rz.sin(), ry.sin(), 0.0],
            [rx.cos()*rz.sin() + rx.sin()*ry.sin()*rz.cos(), sy*(rx.cos()*rz.cos() - rx.sin()*ry.sin()*rz.sin()), -rx.sin()*ry.cos(), 0.0],
            [rx.sin()*rz.sin() - rx.cos()*ry.sin()*rz.cos(), rx.sin()*rz.cos() + rx.cos()*ry.sin()*rz.sin(), sz*rx.cos()*ry.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ],
        perspective: perspective
    };

    let draw_parameters_substrate = glium::draw_parameters::DrawParameters {
        line_width: Some(line_width),
        blend: glium::Blend::alpha_blending(),
        smooth: Some(glium::draw_parameters::Smooth::Nicest),
        viewport: Some(viewport),
        ..Default::default()
    };

    // substrate
    target.draw(&vertex_buffer,
              &line_index_buffer,
              program,
              &uniforms_substrate,
              &draw_parameters_substrate)
        .unwrap();

    let draw_parameters_substrate = glium::draw_parameters::DrawParameters {
        point_size: Some(point_size),
        viewport: Some(viewport),
        ..Default::default()
    };

    let point_index_buffer = glium::index::NoIndices(PrimitiveType::Points);
    target.draw(&vertex_buffer,
              &point_index_buffer,
              program,
              &uniforms_substrate,
              &draw_parameters_substrate)
        .unwrap();
}
