/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
  //modified by zeng @Beihang University on 2018/07
 *
 * Author: Guido Kanschat, Texas A&M University, 2009
 */


// The first few files have already been covered in previous examples and will
// thus not be further commented on:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_tools.h>
// Here the discontinuous finite elements are defined. They are used in the
// same way as all other finite elements, though -- as you have seen in
// previous tutorial programs -- there isn't much user interaction with finite
// element classes at all: they are passed to <code>DoFHandler</code> and
// <code>FEValues</code> objects, and that is about it.
#include <deal.II/fe/fe_dgq.h>
// We are going to use the simplest possible solver, called Richardson
// iteration, that represents a simple defect correction. This, in combination
// with a block SSOR preconditioner (defined in precondition_block.h), that
// uses the special block matrix structure of system matrices arising from DG
// discretizations.
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
// We are going to use gradients as refinement indicator.
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>

// Here come the new include files for using the MeshWorker framework. The
// first contains the class MeshWorker::DoFInfo, which provides local
// integrators with a mapping between local and global degrees of freedom. It
// stores the results of local integrals as well in its base class
// Meshworker::LocalResults.  In the second of these files, we find an object
// of type MeshWorker::IntegrationInfo, which is mostly a wrapper around a
// group of FEValues objects. The file <tt>meshworker/simple.h</tt> contains
// classes assembling locally integrated data into a global system containing
// only a single matrix. Finally, we will need the file that runs the loop
// over all mesh cells and faces.
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

// Like in all programs, we finish this section by including the needed C++
// headers and declaring we want to use objects in the dealii namespace
// without prefix.
#include <iostream>
#include <fstream>
#include <iomanip>


namespace Step12
{
  using namespace dealii;
  namespace PrimalSolverData  //contain the b.cs and rhs for the primal solver
  {
  template <int dim>
  class BoundaryValues:  public Function<dim>
  {
  public:
    BoundaryValues () {};
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component=0) const;
  };
  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim> > &points,
                                       std::vector<double> &values,
                                       const unsigned int) const
  {
    Assert(values.size()==points.size(),
           ExcDimensionMismatch(values.size(),points.size()));

    for (unsigned int i=0; i<values.size(); ++i)
      {
        if (points[i](0)>=0.125 && points[i](0)<=0.75)
          values[i]=1.;
        else
          values[i]=0.;
      }
  }
  template<int dim>
  class RightHandSide: public Function<dim>
  {
  public:
   RightHandSide():Function<dim>(){};
   virtual double value (const Point<dim>& p,
                         const unsigned int component=0) const;
  };
  template<int dim>
  double RightHandSide<dim>::value(const Point<dim>& p,
                                   const unsigned int /*component*/)  const 
  {
   double return_value = 0.;
   //for (unsigned int i=0;i<dim;++i)
   //     return_value += std::pow(p(i),2.);   //rhs function is: f=x^2+y^2
   return return_value;          //the equation has no rhs. 
  }
  }  //end of the PrimalSolverData namespace




  namespace AdvectionSolver
  {
   template<int dim>
   class Base
   {
   public:
           Base();
	   //Base(Triangulation<dim>& coarse_grid);
	   virtual ~Base();
           virtual void initialize_problem()=0;
	   virtual void solve_problem()=0;
           virtual void output_results(unsigned int cycle) const=0;
	   virtual void refine_grid()=0;
           virtual unsigned int n_dofs() const=0;
           virtual unsigned int n_active_cells() const=0;
   protected:
       //const SmartPointer<Triangulation<dim>> triangulation;
       Triangulation<dim> triangulation;
   };
   template<int dim>
   Base<dim>::Base()
   //Base<dim>::Base(Triangulation<dim>& coarse_grid):triangulation(&coarse_grid)
   {}
   template<int dim>
   Base<dim>::~Base()
   {}



   template<int dim>
   class Solver:public virtual Base<dim>
   {
   public:
       //Solver(Triangulation<dim>  &triangulation,FE_DGQ<dim>         &fe);
       Solver(const FiniteElement<dim>& fe,
              const Quadrature<dim>&    quadrature,
              const Quadrature<dim-1>&  face_quadrature);
       virtual ~Solver();
       virtual void solve_problem();
       virtual unsigned int n_dofs() const;
       virtual unsigned int n_active_cells() const;

   protected:
       const MappingQ1<dim> mapping;        //this const member must be initialized in the constructor
       const FiniteElement<dim>&       fe;   //define a reference to the fe passed to Solver: FE_DGQ
       const Quadrature<dim>   quadrature;
       const Quadrature<dim-1> face_quadrature;
       DoFHandler<dim>   dof_handler;
       Vector<double>    solution;
                                           //these virtual functions inherited from the base class
       void setup_system();                 //can only be called in a public member function
       virtual void assemble_system()=0;  //remain to complete
       typedef MeshWorker::DoFInfo<dim>          DoFInfo;
       typedef MeshWorker::IntegrationInfo<dim>  CellInfo;       
       virtual void integrate_cell_term(DoFInfo& dinfo,CellInfo& info)=0;
       virtual void integrate_boundary_term(DoFInfo& dinfo,CellInfo& info)=0;
       virtual void integrate_face_term(DoFInfo& dinfo1,DoFInfo& dinfo2,CellInfo& info1,CellInfo& info2)=0;
 
       struct LinearSystem{
                   LinearSystem();
		   void reinit(DoFHandler<dim>& dof_handler);
		   void solve(Vector<double>& solution, const FiniteElement<dim>& fe) const;
		   SparsityPattern           sparsity_pattern;
		   SparseMatrix<double>      system_matrix;
	           Vector<double>            right_hand_side;
	   };
       LinearSystem linear_system;  //need default constructor, similar to dof_handler

   };
   //constructor of Solver 
   template<int dim>
   Solver<dim>::Solver(const FiniteElement<dim>& fe,
                       const Quadrature<dim>&    quadrature,
                       const Quadrature<dim-1>&  face_quadrature)
                :
                Base<dim>(),
                mapping(),
                fe(fe),       //here (fe) should be a dg_fe.
                quadrature(quadrature),
                face_quadrature(face_quadrature),
                dof_handler(Base<dim>::triangulation) //initialize the member objects.
   {}
   template<int dim>
   Solver<dim>::~Solver(){
        dof_handler.clear();
   }
/*
   template<int dim>
   Solver<dim>::Solver(Triangulation<dim>  &triangulation,
	               FE_DGQ<dim>         &fe)
                :
                Base<dim>(triangulation),fe(fe),dof_handler(triangulation)
   {}
*/
   template<int dim>
   Solver<dim>::LinearSystem::LinearSystem()    //default constructor
   {}
   template<int dim>
   void
   Solver<dim>::LinearSystem::reinit(DoFHandler<dim>& dof_handler){
       DynamicSparsityPattern dsp(dof_handler.n_dofs());
       DoFTools::make_flux_sparsity_pattern (dof_handler, dsp);
       sparsity_pattern.copy_from(dsp);
       system_matrix.reinit (sparsity_pattern);
       right_hand_side.reinit (dof_handler.n_dofs());
   }
   //the system solver can be modified when needed.
   template<int dim>
   void
   Solver<dim>::LinearSystem::solve(Vector<double>& solution, const FiniteElement<dim>& fe) const {
       SolverControl                                 solver_control(1000,1e-12);
       SolverRichardson<>                            solver(solver_control);
       PreconditionBlockSSOR<SparseMatrix<double>>   preconditioner;
       preconditioner.initialize(system_matrix,fe.dofs_per_cell);
       solver.solve(system_matrix,solution,right_hand_side,preconditioner);
   }
   
   template<int dim>
   void
   Solver<dim>::setup_system(){
       dof_handler.distribute_dofs(fe);
       solution.reinit(dof_handler.n_dofs());
       linear_system.reinit(dof_handler);  //similar to dof_handler.distribute_dofs(fe)
   }
   template<int dim>
   void
   Solver<dim>::solve_problem(){    //this function will be called as "dg_method.solve_problem()"
       setup_system();
       assemble_system();
       linear_system.solve(solution,fe);
   }
   template<int dim>
   unsigned int
   Solver<dim>::n_dofs()const{
       return dof_handler.n_dofs();
   }
   template<int dim>
   unsigned int
   Solver<dim>::n_active_cells() const{
       return Base<dim>::triangulation.n_active_cells();
   }



//the primal solver
   template<int dim>
   class PrimalSolver:public Solver<dim>{
   public:
/*
       PrimalSolver(Triangulation<dim>  &triangulation,
	                FE_DGQ<dim>         &fe,
					Function<dim> &boundary_function
					Function<dim>  &rhs_function);
*/
       PrimalSolver(const FiniteElement<dim>& fe,
                    const Quadrature<dim>&    quadrature,
                    const Quadrature<dim-1>&  face_quadrature);
       virtual void initialize_problem();                     //inherite from the base class
       virtual void output_results(unsigned int cycle) const;
       virtual void refine_grid();
   protected:
       PrimalSolverData::RightHandSide<dim>   rhs_function;       //specified by user
       PrimalSolverData::BoundaryValues<dim>  boundary_function;
   private:
       virtual void assemble_system();
       typedef MeshWorker::DoFInfo<dim>          DoFInfo;
       typedef MeshWorker::IntegrationInfo<dim>  CellInfo;       
       virtual void integrate_cell_term(DoFInfo& dinfo,CellInfo& info);
       virtual void integrate_boundary_term(DoFInfo& dinfo,CellInfo& info);
       virtual void integrate_face_term(DoFInfo& dinfo1,DoFInfo& dinfo2,CellInfo& info1,CellInfo& info2);
   };
   //constructor of primal solver, it's called as 
   //"PrimalSolver<2> dg_method"(triangulation,fe,boundary_function,rhs_function)
/*
   template<int dim>
   PrimalSolver(Triangulation<dim> &triangulation,
                FE_DGQ<dim>        &fe,
                Function<dim>      &boundary_function,
                Function<dim>      &rhs_function)
                :
                Base<dim>(triangulation),
                Solver<dim>(triangulation,fe),
                boundary_function(),
                rhs_function()
   {}
*/
   template<int dim>
   PrimalSolver<dim>::PrimalSolver(const FiniteElement<dim>& fe,
                                   const Quadrature<dim>&    quadrature,
                                   const Quadrature<dim-1>&  face_quadrature)
                      :
                      Base<dim>(),
                      Solver<dim>(fe,quadrature,face_quadrature)
   {}
   template<int dim>
   void 
   PrimalSolver<dim>::output_results(unsigned int cycle) const{
    // Write the grid in eps format.
    std::string filename = "grid-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".eps";
    deallog << "Writing grid to <" << filename << ">" << std::endl;
    std::ofstream eps_output (filename.c_str());

    GridOut grid_out;
    grid_out.write_eps (Base<dim>::triangulation, eps_output);

    // Output of the solution in gnuplot format.
    filename = "sol-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".gnuplot";
    deallog << "Writing solution to <" << filename << ">" << std::endl;
    std::ofstream gnuplot_output (filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (Solver<dim>::dof_handler);
    data_out.add_data_vector (Solver<dim>::solution, "u");

    data_out.build_patches ();

    data_out.write_gnuplot(gnuplot_output);
   }
   
   template<int dim>
   void
   PrimalSolver<dim>::refine_grid(){
    Vector<float> gradient_indicator (Base<dim>::triangulation.n_active_cells());

    // Now the approximate gradients are computed
    DerivativeApproximation::approximate_gradient (Solver<dim>::mapping,
                                                   Solver<dim>::dof_handler,
                                                   Solver<dim>::solution,
                                                   gradient_indicator);

    // and they are cell-wise scaled by the factor $h^{1+d/2}$
    typename DoFHandler<dim>::active_cell_iterator
    cell = Solver<dim>::dof_handler.begin_active(),
    endc = Solver<dim>::dof_handler.end();
    for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
      gradient_indicator(cell_no)*=std::pow(cell->diameter(), 1+1.0*dim/2);

    // Finally they serve as refinement indicator.
    GridRefinement::refine_and_coarsen_fixed_number (Base<dim>::triangulation,
                                                     gradient_indicator,
                                                     0.3, 0.1);

    Base<dim>::triangulation.execute_coarsening_and_refinement ();
   }
   
   template<int dim>
   void
   PrimalSolver<dim>::initialize_problem(){
    const Point<dim> bottom_left = Point<dim>();
    const Point<dim> upper_right = Point<dim>(2.,1.);   //used to specify the domain

    std::vector<unsigned int> repetitions;   //used to subdivide the original domain
    repetitions.push_back (8);
    if (dim>=2)
       repetitions.push_back (4);
    //repetitions is a vector with 2 elements of value 8 and 4.
    
    GridGenerator::subdivided_hyper_rectangle(Base<dim>::triangulation,repetitions,bottom_left,upper_right);  //creat a rectangle triangulation, p1/p2 specify the domain
   }     

   template<int dim>
   void
   PrimalSolver<dim>::assemble_system(){
       MeshWorker::IntegrationInfoBox<dim>   info_box;
       const unsigned int n_gauss_points = Solver<dim>::quadrature.size();
       info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);

       info_box.initialize_update_flags();
       UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
       info_box.add_update_flags(update_flags,true,true,true,true);

       info_box.initialize(Solver<dim>::fe,Solver<dim>::mapping);

       MeshWorker::DoFInfo<dim>              dof_info(Solver<dim>::dof_handler);
       MeshWorker::Assembler::SystemSimple<SparseMatrix<double>,Vector<double>>   assembler;
       assembler.initialize(Solver<dim>::linear_system.system_matrix,
                            Solver<dim>::linear_system.right_hand_side);

       MeshWorker::loop<dim,dim,MeshWorker::DoFInfo<dim>,MeshWorker::IntegrationInfoBox<dim>>
       (Solver<dim>::dof_handler.begin_active(),Solver<dim>::dof_handler.end(),
        dof_info,info_box,
        std::bind(&PrimalSolver<dim>::integrate_cell_term,this,std::placeholders::_1,std::placeholders::_2),
        std::bind(&PrimalSolver<dim>::integrate_boundary_term,this,std::placeholders::_1,std::placeholders::_2),
        std::bind(&PrimalSolver<dim>::integrate_face_term,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4),
        assembler);
    }
  template<int dim>
  void PrimalSolver<dim>::integrate_cell_term(DoFInfo &dinfo,CellInfo &info){
        const FEValuesBase<dim> &fe_v = info.fe_values();
        FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
        Vector<double> &local_vector=dinfo.vector(0).block(0);   //I add this vector to store the rhs_function term
        const std::vector<double> &JxW = fe_v.get_JxW_values ();

    // With these objects, we continue local integration like always. First,
    // we loop over the quadrature points and compute the advection vector in
    // the current point.
        for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
          {
	    Point<dim> beta;
	    if(fe_v.quadrature_point(point)(0)<1){
	        beta(0) = fe_v.quadrature_point(point)(1); 
	        beta(1) = 1.-fe_v.quadrature_point(point)(0);
	    }
	    else{
	        beta(0) = 2.-fe_v.quadrature_point(point)(1);
	        beta(1) = fe_v.quadrature_point(point)(0)-1.;
	    } 
	    beta /= beta.norm();

        // We solve a homogeneous equation, thus no right hand side shows up
        // in the cell term.  What's left is integrating the matrix entries.
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                local_matrix(i,j) -= beta*fe_v.shape_grad(i,point)*
                                     fe_v.shape_value(j,point) *
                                     JxW[point];
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)  //this is the right_hand_side
                local_vector(i) += fe_v.shape_value(i,point)*
                                   rhs_function.value(fe_v.quadrature_point(point))*
                                   JxW[point];   //return position of the point-th point in real space. 
          }
    }
    template <int dim>
    void PrimalSolver<dim>::integrate_boundary_term (DoFInfo &dinfo, CellInfo &info){
        const FEValuesBase<dim> &fe_v = info.fe_values();
        FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
        Vector<double> &local_vector = dinfo.vector(0).block(0);

        const std::vector<double> &JxW = fe_v.get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();

        std::vector<double> g(fe_v.n_quadrature_points);
        boundary_function.value_list (fe_v.get_quadrature_points(), g);

        for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
          {
            Point<dim> beta;
            if(fe_v.quadrature_point(point)(0)<1){
                beta(0) = fe_v.quadrature_point(point)(1); 
                beta(1) = 1.-fe_v.quadrature_point(point)(0);
            }
            else{
                beta(0) = 2.-fe_v.quadrature_point(point)(1);
                beta(1) = fe_v.quadrature_point(point)(0)-1.;
            } 
            beta /= beta.norm();

            const double beta_n=beta * normals[point];
            if (beta_n>0)
              for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                  local_matrix(i,j) += beta_n *
                                       fe_v.shape_value(j,point) *
                                       fe_v.shape_value(i,point) *
                                       JxW[point];
            else
              for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                local_vector(i) -= beta_n *
                                   g[point] *
                                   fe_v.shape_value(i,point) *
                                   JxW[point];
          }
    }
  template <int dim>
  void PrimalSolver<dim>::integrate_face_term (DoFInfo &dinfo1,DoFInfo &dinfo2,
                                               CellInfo &info1,CellInfo &info2)
  {
    const FEValuesBase<dim> &fe_v = info1.fe_values();
    const FEValuesBase<dim> &fe_v_neighbor = info2.fe_values();
    FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0,false).matrix;
    FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0,true).matrix;
    FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0,true).matrix;
    FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0,false).matrix;

    const std::vector<double> &JxW = fe_v.get_JxW_values ();
    const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
        Point<dim> beta;
        if(fe_v.quadrature_point(point)(0)<1){
            beta(0) = fe_v.quadrature_point(point)(1); 
            beta(1) = 1.-fe_v.quadrature_point(point)(0);
        }
        else{
            beta(0) = 2.-fe_v.quadrature_point(point)(1);
            beta(1) = fe_v.quadrature_point(point)(0)-1.;
        } 
        beta /= beta.norm();

        const double beta_n=beta * normals[point];
        if (beta_n>0)
          {
            // This term we've already seen:
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                u1_v1_matrix(i,j) += beta_n *
                                     fe_v.shape_value(j,point) *
                                     fe_v.shape_value(i,point) *
                                     JxW[point];

            // We additionally assemble the term $(\beta\cdot n u,\hat
            // v)_{\partial \kappa_+}$,
            for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                u1_v2_matrix(k,j) -= beta_n *
                                     fe_v.shape_value(j,point) *
                                     fe_v_neighbor.shape_value(k,point) *
                                     JxW[point];
          }
        else
          {
            // This one we've already seen, too:
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                u2_v1_matrix(i,l) += beta_n *
                                     fe_v_neighbor.shape_value(l,point) *
                                     fe_v.shape_value(i,point) *
                                     JxW[point];

            // And this is another new one: $(\beta\cdot n \hat u,\hat
            // v)_{\partial \kappa_-}$:
            for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
              for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                u2_v2_matrix(k,l) -= beta_n *
                                     fe_v_neighbor.shape_value(l,point) *
                                     fe_v_neighbor.shape_value(k,point) *
                                     JxW[point];
          }
      }
  }
 }  //the namespace AdvectionSolver end here

  namespace DualSolverData   //contain the b.cs and rhs of the dualsolver
  {
   template <int dim>
   class BoundaryValues:  public Function<dim>
   {
   public:
     BoundaryValues () {};
     virtual void value_list (const std::vector<Point<dim> > &points,
                              std::vector<double> &values,
                              const unsigned int component=0) const;
   };
   template <int dim>
   void BoundaryValues<dim>::value_list(const std::vector<Point<dim> > &points,
                                        std::vector<double> &values,
                                        const unsigned int) const
   {
     Assert(values.size()==points.size(),
            ExcDimensionMismatch(values.size(),points.size()));

     for (unsigned int i=0; i<values.size(); ++i)
       {
         if (points[i](1)>=0.25 && points[i](1)<=1.)
           values[i]=exp(pow(3./8,-2)-pow((points[i](1)-5./8)*(points[i](1)-5./8)-3./8,-2));
         else
           values[i]=0.;
       }
   }
   template<int dim>
   class RightHandSide: public Function<dim>
   {
   public:
    RightHandSide():Function<dim>(){};
    virtual double value (const Point<dim>& p,
                          const unsigned int component=0) const;
   };
   template<int dim>
   double RightHandSide<dim>::value(const Point<dim>& p,
                                    const unsigned int /*component*/)  const 
   {
    double return_value = 0.;
    //for (unsigned int i=0;i<dim;++i)
    //     return_value += std::pow(p(i),2.);   //rhs function is: f=x^2+y^2
    return return_value;          //the equation has no rhs. 
   }
  }   //end of the DualSolverData namespace




  //Dual Solver
  namespace AdvectionSolver
  {
   template <int dim>
   class DualSolver:public Solver<dim>{
   public:
      DualSolver(const FiniteElement<dim>& fe,
                 const Quadrature<dim>&    quadrature,
                 const Quadrature<dim-1>&  face_quadrature);
      virtual void initialize_problem();                     //inherite from the base class
      //virtual void output_results(unsigned int cycle) const;
   protected:
      DualSolverData::RightHandSide<dim>   rhs_function;       //specified by user
      DualSolverData::BoundaryValues<dim>  boundary_function;
   private: 
      virtual void assemble_system();
      typedef MeshWorker::DoFInfo<dim>          DoFInfo;
      typedef MeshWorker::IntegrationInfo<dim>  CellInfo;       
      virtual void integrate_cell_term(DoFInfo& dinfo,CellInfo& info);
      virtual void integrate_boundary_term(DoFInfo& dinfo,CellInfo& info);
      virtual void integrate_face_term(DoFInfo& dinfo1,DoFInfo& dinfo2,CellInfo& info1,CellInfo& info2);
   };
   template<int dim>
   DualSolver<dim>::DualSolver(const FiniteElement<dim>& fe,
                               const Quadrature<dim>&    quadrature,
                               const Quadrature<dim-1>&  face_quadrature)
                    :
                    Base<dim>(),
                    Solver<dim>(fe,quadrature,face_quadrature)
   {}
   template<int dim>
   void
   DualSolver<dim>::initialize_problem(){
      //do nothing in this class
   }
   template<int dim>
   void 
   DualSolver<dim>::assemble_system(){
       MeshWorker::IntegrationInfoBox<dim>   info_box;   //provide the FeValues needed to do integration
       const unsigned int n_gauss_points = Solver<dim>::quadrature.size();
       info_box.initialize_gauss_quadrature(n_gauss_points,n_gauss_points,n_gauss_points);

       info_box.initialize_update_flags();
       UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
       info_box.add_update_flags(update_flags,true,true,true,true);

       info_box.initialize(Solver<dim>::fe,Solver<dim>::mapping); //till now the info_box is ready

       MeshWorker::DoFInfo<dim>         dof_info(Solver<dim>::dof_handler); //used to store the results
       MeshWorker::Assembler::SystemSimple<SparseMatrix<double>,Vector<double>>   assembler;
       assembler.initialize(Solver<dim>::linear_system.system_matrix,
                            Solver<dim>::linear_system.right_hand_side);

       MeshWorker::loop<dim,dim,MeshWorker::DoFInfo<dim>,MeshWorker::IntegrationInfoBox<dim>>
       (Solver<dim>::dof_handler.begin_active(),Solver<dim>::dof_handler.end(),
        dof_info,info_box,
        std::bind(&DualSolver<dim>::integrate_cell_term,this,std::placeholders::_1,std::placeholders::_2),
        std::bind(&DualSolver<dim>::integrate_boundary_term,this,std::placeholders::_1,std::placeholders::_2),
        std::bind(&DualSolver<dim>::integrate_face_term,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4),
        assembler);
   }
   template<int dim>
   void
   DualSolver<dim>::integrate_cell_term(DoFInfo& dinfo,CellInfo& info)
   {
      const FEValuesBase<dim>& fe_v = info.fe_values();
      FullMatrix<double>& local_matrix = dinfo.matrix(0).matrix;
      const std::vector<double>& JxW = fe_v.get_JxW_values();
      for(unsigned int point=0;point<fe_v.n_quadrature_points;++point){
         Point<dim> alpha;   //alpha is opposite to beta of the primal problem
	 if(fe_v.quadrature_point(point)(0)<1){
	     alpha(0) = -fe_v.quadrature_point(point)(1); 
	     alpha(1) = -(1.-fe_v.quadrature_point(point)(0));
	 }
	 else{
	     alpha(0) = -(2.-fe_v.quadrature_point(point)(1));
	     alpha(1) = -(fe_v.quadrature_point(point)(0)-1.);
	 } 
	 alpha /= alpha.norm();
         for(unsigned int i=0;i<fe_v.dofs_per_cell;++i)
            for(unsigned int j=0;j<fe_v.dofs_per_cell;++j)
               local_matrix(i,j)-=alpha*fe_v.shape_grad(i,point)*fe_v.shape_value(j,point)*JxW[point];
      } 
   }
   template<int dim>
   void
   DualSolver<dim>::integrate_boundary_term(DoFInfo& dinfo,CellInfo& info){
      const FEValuesBase<dim> &fe_v = info.fe_values();
      FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
      Vector<double> &local_vector = dinfo.vector(0).block(0);

      const std::vector<double> &JxW = fe_v.get_JxW_values ();
      const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();

      std::vector<double> psi(fe_v.n_quadrature_points);  //use psi to store the b.c of the dual prob
      boundary_function.value_list (fe_v.get_quadrature_points(), psi);
      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
        {
          Point<dim> alpha;
          if(fe_v.quadrature_point(point)(0)<1){
              alpha(0) = -fe_v.quadrature_point(point)(1); 
              alpha(1) = -(1.-fe_v.quadrature_point(point)(0));
          }
          else{
              alpha(0) = -(2.-fe_v.quadrature_point(point)(1));
              alpha(1) = -(fe_v.quadrature_point(point)(0)-1.);
          } 
          alpha /= alpha.norm();

          const double alpha_n=alpha * normals[point];
          if (alpha_n>0)
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                local_matrix(i,j) += alpha_n *
                                     fe_v.shape_value(j,point) *
                                     fe_v.shape_value(i,point) *
                                     JxW[point];
          else
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              local_vector(i) -= -psi[point] *                              //here alpha_n*g=psi
                                 fe_v.shape_value(i,point) *
                                 JxW[point];
        }
   }
   template<int dim>
   void
   DualSolver<dim>::integrate_face_term(DoFInfo& dinfo1,DoFInfo& dinfo2,
                                        CellInfo& info1,CellInfo& info2)
   {
    const FEValuesBase<dim> &fe_v = info1.fe_values();
    const FEValuesBase<dim> &fe_v_neighbor = info2.fe_values();
    FullMatrix<double> &z1_w1_matrix = dinfo1.matrix(0,false).matrix;
    FullMatrix<double> &z2_w1_matrix = dinfo1.matrix(0,true).matrix;
    FullMatrix<double> &z1_w2_matrix = dinfo2.matrix(0,true).matrix;
    FullMatrix<double> &z2_w2_matrix = dinfo2.matrix(0,false).matrix;

    const std::vector<double> &JxW = fe_v.get_JxW_values ();
    const std::vector<Tensor<1,dim> > &normals = fe_v.get_all_normal_vectors ();

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
        Point<dim> alpha;
        if(fe_v.quadrature_point(point)(0)<1){
            alpha(0) = -fe_v.quadrature_point(point)(1); 
            alpha(1) = -(1.-fe_v.quadrature_point(point)(0));
        }
        else{
            alpha(0) = -(2.-fe_v.quadrature_point(point)(1));
            alpha(1) = -(fe_v.quadrature_point(point)(0)-1.);
        } 
        alpha /= alpha.norm();

        const double alpha_n=alpha * normals[point];
        if (alpha_n>0)
          {
            // This term we've already seen:
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                z1_w1_matrix(i,j) += alpha_n *
                                     fe_v.shape_value(j,point) *
                                     fe_v.shape_value(i,point) *
                                     JxW[point];

            // We additionally assemble the term $(\beta\cdot n u,\hat
            // v)_{\partial \kappa_+}$,
            for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
              for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                z1_w2_matrix(k,j) -= alpha_n *
                                     fe_v.shape_value(j,point) *
                                     fe_v_neighbor.shape_value(k,point) *
                                     JxW[point];
          }
        else
          {
            // This one we've already seen, too:
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
              for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                z2_w1_matrix(i,l) += alpha_n *
                                     fe_v_neighbor.shape_value(l,point) *
                                     fe_v.shape_value(i,point) *
                                     JxW[point];

            // And this is another new one: $(\beta\cdot n \hat u,\hat
            // v)_{\partial \kappa_-}$:
            for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
              for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
                z2_w2_matrix(k,l) -= alpha_n *
                                     fe_v_neighbor.shape_value(l,point) *
                                     fe_v_neighbor.shape_value(k,point) *
                                     JxW[point];
          }
      }
  
   }
   
   enum RefinementCriterion
   {
     dual_weighted_error_estimator,
     global_refinement,
     kelly_indicator,
     derivative 
   }; 

   //the WeightedResidual class
   template<int dim>
   class WeightedResidual:public PrimalSolver<dim>,public DualSolver<dim>{
   public:
      WeightedResidual(const FiniteElement<dim>& primal_fe,
                       const FiniteElement<dim>& dual_fe,
                       const Quadrature<dim>&    quadrature,
                       const Quadrature<dim-1>&  face_quadrature);
      virtual void initialize_problem();
      virtual void solve_problem();
      virtual void output_results(unsigned int cycle);
      virtual void refine_grid();
      virtual unsigned int n_dofs() const;
      virtual unsigned int n_active_cells() const;
      double return_functional() const;
   private:
      std::clock_t                start, timer;
      typedef typename std::map<typename DoFHandler<dim>::face_iterator,double> FaceIntegrals;
      typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
      struct CellData{
         FEValues<dim> fe_values;
         std::vector<double> cell_residual;
         std::vector<double> rhs_values;
         std::vector<double> dual_weights;
         const SmartPointer<const Function<dim>> right_hand_side;   //this pointer always point to the rhs function of the primal_solver
         typename std::vector<Tensor<1,dim>> cell_grads;  //tensor of rank 1 (i.e. a vector with dim components), why need typename??
         CellData(const FiniteElement<dim>& fe,
                  const Quadrature<dim>&    quadrature,
                  const Function<dim>&      right_hand_side);
      };
      //store several data structure in face_data. here the fe_face_values_cell etc. are just interfaces used to reinit data on every cell. and face_values etc. store the "real" data needed for computation of face_term_error.
      struct FaceData{
         FEFaceValues<dim> fe_face_values_cell;
         FEFaceValues<dim> fe_face_values_neighbor;
         FESubfaceValues<dim> fe_subface_values_cell;
         FESubfaceValues<dim> fe_subface_values_neighbor;
         std::vector<double> face_residual;
         std::vector<double> dual_weights;
         std::vector<double> face_values;
         std::vector<double> neighbor_values;
         FaceData(const FiniteElement<dim>& fe,
                  const Quadrature<dim-1>&  face_quadrature);
      };
      void estimate_error(Vector<float>& error_indicators) const;
      void integrate_over_cell(const SynchronousIterators<std_cxx11::tuple<
                               active_cell_iterator,Vector<float>::iterator>>&  cell_and_error,
                               CellData&                                        cell_data,   
                               const Vector<double>&                            primal_solution,
                               const Vector<double>&                            dual_weights) const;
      void integrate_over_face(active_cell_iterator& cell,
                               const unsigned int    face_no,
                               FaceData&             face_data,
                               Vector<double>&       primal_solution,
                               Vector<double>&       dual_weights, 
                               FaceIntegrals&        face_integrals) const;
   };
   template<int dim>
   WeightedResidual<dim>::WeightedResidual(const FiniteElement<dim>& primal_fe,
                                           const FiniteElement<dim>& dual_fe,
                                           const Quadrature<dim>&    quadrature,
                                           const Quadrature<dim-1>&  face_quadrature)
                          :
                          Base<dim>(),
                          PrimalSolver<dim>(primal_fe,quadrature,face_quadrature),
                          DualSolver<dim>(dual_fe,quadrature,face_quadrature)
   {}
   //constructor of CellData
   template<int dim>
   WeightedResidual<dim>::CellData::
   CellData(const FiniteElement<dim>& fe,
            const Quadrature<dim>&    quadrature,
            const Function<dim>&      right_hand_side):
      fe_values(fe,quadrature,update_values|
                              update_gradients|
                              update_quadrature_points|
                              update_JxW_values),
      cell_residual(quadrature.size()),
      rhs_values(quadrature.size()),
      dual_weights(quadrature.size()),
      right_hand_side(&right_hand_side),
      cell_grads(quadrature.size())
   {}
   //constructor of FaceData
   template<int dim>
   WeightedResidual<dim>::FaceData::
   FaceData(const FiniteElement<dim>& fe,
            const Quadrature<dim-1>&  face_quadrature):
      fe_face_values_cell(fe,face_quadrature,
                          update_values|
                          update_quadrature_points|
                          update_JxW_values|
                          update_normal_vectors),
      fe_face_values_neighbor(fe,face_quadrature,
                              update_values|
                              update_quadrature_points),
      fe_subface_values_cell(fe,face_quadrature,
                             update_values|
                             update_quadrature_points|
                             update_JxW_values|
                             update_normal_vectors),
      fe_subface_values_neighbor(fe,face_quadrature,
                                 update_values|
                                 update_quadrature_points),
      face_residual(face_quadrature.size()),
      dual_weights(face_quadrature.size()),
      face_values(face_quadrature.size()),
      neighbor_values(face_quadrature.size())
   {/*???*/}
   template<int dim>
   void
   WeightedResidual<dim>::initialize_problem()
   {
      start = std::clock();
      PrimalSolver<dim>::initialize_problem();  //only use PrimalSolver to edit the triangulation
      //DualSolver<dim>::initialize_problem();   
   }
   template<int dim>
   void 
   WeightedResidual<dim>::solve_problem(){
      //
      PrimalSolver<dim>::solve_problem();
      DualSolver<dim>::solve_problem();
   }
   template<int dim>
   void
   WeightedResidual<dim>::output_results(unsigned int cycle){
      // Write the grid in eps format.
    std::string filename = "primal_grid-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".eps";
    deallog << "Writing grid to <" << filename << ">" << std::endl;
    std::ofstream eps_output (filename.c_str());

    GridOut grid_out;
    grid_out.write_eps (Base<dim>::triangulation, eps_output);

    //interpolate the dual_solution to primal fe space
    Vector<double> dual_solution(PrimalSolver<dim>::dof_handler.n_dofs());
    FETools::interpolate(DualSolver<dim>::dof_handler,
                         DualSolver<dim>::solution,
                         PrimalSolver<dim>::dof_handler,
                         dual_solution);
    // Output of the solution in gnuplot format.
    filename = "sol-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError());

    filename += ".gnuplot";
    deallog << "Writing solution to <" << filename << ">" << std::endl;
    std::ofstream gnuplot_output (filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (PrimalSolver<dim>::dof_handler);
    data_out.add_data_vector (PrimalSolver<dim>::solution, "primal_solution");
    data_out.add_data_vector (dual_solution,"dual_solution");  //dual_solutino needs to be interpolated firstly
    data_out.build_patches ();

    data_out.write_gnuplot(gnuplot_output);

    //write out dofs----time----true error in the target functional successively to the same file
    std::ofstream curve_graph("curve_graph.txt", std::ios::app);      
    timer = std::clock();
    double current_time = (double)(timer-start)/CLOCKS_PER_SEC;

    if(cycle == 0)
       curve_graph<<"dofs"
                  <<std::setw(30)<<"J(e)"
                  <<std::setw(30)<<"time"<<std::endl;                  
    else
       curve_graph<<n_dofs()
                  <<std::setw(30)<<std::setprecision(15)<<(return_functional()-0.192808353547215)
                  <<std::setw(30)<<std::setprecision(15)<<current_time<<std::endl;
   }
   template<int dim>
   void
   WeightedResidual<dim>::refine_grid(){   
      Vector<float> error_indicators(PrimalSolver<dim>::n_active_cells());

      //You have to choose here which refinement_criterion to use in this code, choices include:
        /****************************
        dual_weighted_error_estimator,
        global_refinement,
        kelly_indicator,
        derivative
        ****************************/ 
      const RefinementCriterion   refinement_criterion = dual_weighted_error_estimator;  
      
      switch(refinement_criterion)
        {
         // use the dual weighted error estimator as indicator to execute the refinement process
         case dual_weighted_error_estimator:
         {
            estimate_error(error_indicators);
            for(Vector<float>::iterator i=error_indicators.begin();i!=error_indicators.end();++i)
               *i = std::fabs(*i);
            GridRefinement::refine_and_coarsen_fixed_number(this->triangulation,error_indicators,0.3,0.1);
            this->triangulation.execute_coarsening_and_refinement();
            break;
         }
         case global_refinement:
         {
            this->triangulation.refine_global(1);
            break;
         }
         case kelly_indicator:
         {
            KellyErrorEstimator<dim>::estimate(PrimalSolver<dim>::dof_handler,
                                               QGauss<dim-1>(3),
                                               typename FunctionMap<dim>::type(),
                                               PrimalSolver<dim>::solution,
                                               error_indicators);
            GridRefinement::refine_and_coarsen_fixed_number(this->triangulation,error_indicators,0.3,0.1);
            this->triangulation.execute_coarsening_and_refinement();
            break;
         }
         case derivative:
         {
            PrimalSolver<dim>::refine_grid();
            break;
         }
        }
   }
   template<int dim>
   unsigned int
   WeightedResidual<dim>::n_dofs() const{
      return PrimalSolver<dim>::n_dofs();   
   }
   template<int dim>
   unsigned int
   WeightedResidual<dim>::n_active_cells() const{
      return PrimalSolver<dim>::n_active_cells();
   }
   template<int dim>
   double
   WeightedResidual<dim>::return_functional()const{
      double J = 0;  //the functional
      DualSolverData::BoundaryValues<dim>  boundary_function;
      FEFaceValues<dim> fe_boundary_face(PrimalSolver<dim>::fe,
                                         PrimalSolver<dim>::face_quadrature,
                                         update_values|
                                         update_quadrature_points|
                                         update_JxW_values);
      typename DoFHandler<dim>::active_cell_iterator 
                                cell = PrimalSolver<dim>::dof_handler.begin_active(),
                                endc = PrimalSolver<dim>::dof_handler.end();
      for(;cell!=endc;++cell)
         for(unsigned int face_no=0;face_no<GeometryInfo<dim>::faces_per_cell;++face_no){
            if(cell->face(face_no)->at_boundary()==false)
               continue;
            else{  
//loop over all boundary face to compute J(u)=(u,psi). return values of vector psi on bounary faces. if the coordinates of the quadrature_point doesn't belongs to {2}x(1/4,1), the psi on this point equals 0.(i.e. has no contribution to the functional)              
               const unsigned int n_q_points = fe_boundary_face.n_quadrature_points;
               std::vector<double> psi(n_q_points);  
               std::vector<double> u(n_q_points);
               //return values of psi and u
               fe_boundary_face.reinit(cell,face_no);             
               fe_boundary_face.get_function_values(PrimalSolver<dim>::solution,u);
               boundary_function.value_list (fe_boundary_face.get_quadrature_points(), psi);
               //compute integral: (psi,u), store in J
               for(unsigned int p=0;p<n_q_points;++p)
                  J += psi[p]*u[p]*fe_boundary_face.JxW(p); 
            }
         }
      return J;
   }// member function end here
   template<int dim>
   void 
   WeightedResidual<dim>::estimate_error(Vector<float>& error_indicators) const{
     //interpolate the primal solution to the dual space
     Vector<double> primal_solution(DualSolver<dim>::dof_handler.n_dofs());
     FETools::interpolate(PrimalSolver<dim>::dof_handler,
                          PrimalSolver<dim>::solution,   //vector_in
                          DualSolver<dim>::dof_handler,
                          primal_solution);              //vector_out
     //the dual weights be in the dual space: z-z_h,need to firstly interpolate z to the primal space
     Vector<double> dual_weights(DualSolver<dim>::dof_handler.n_dofs());
     FETools::interpolation_difference(DualSolver<dim>::dof_handler,
                                       DualSolver<dim>::solution,    //vector_in:z
                                       PrimalSolver<dim>::fe,
                                       dual_weights);                //vector_out:z-zh
     //define the cell iterator(it is bind with error_indicator's iterator)
     typedef std_cxx11::tuple<active_cell_iterator,Vector<float>::iterator> IteratorTuple;
     SynchronousIterators<IteratorTuple>  
     cell_and_error_begin(IteratorTuple(DualSolver<dim>::dof_handler.begin_active(),
                                        error_indicators.begin())),
     cell_and_error_end  (IteratorTuple(DualSolver<dim>::dof_handler.end(),
                                        error_indicators.end()));
     //construct cell_data and face_data, used to store required values on quadrature_points (in dual space)
     CellData cell_data(DualSolver<dim>::fe,
                        DualSolver<dim>::quadrature,
                        PrimalSolver<dim>::rhs_function);//why here need rhs_function of the primal_s??
     FaceData face_data(DualSolver<dim>::fe,
                        DualSolver<dim>::face_quadrature);

     unsigned int present_cell = 0;
     for(SynchronousIterators<IteratorTuple> cell_and_error=cell_and_error_begin;
                                             cell_and_error!=cell_and_error_end;
                                             ++cell_and_error,++present_cell){
        //compute the cell_term, stored in error_indicators
        integrate_over_cell(cell_and_error, cell_data, primal_solution, dual_weights);
       // std::cout<<"cell_term_error on current cell is "
         //        <<*(std_cxx11::get<1>(*cell_and_error))
           //      <<std::endl;

        active_cell_iterator cell = std_cxx11::get<0>(*cell_and_error);

        //define a local map to store face_term(4 values for 2D) on current cell
        FaceIntegrals face_integrals;  //this is a map from face_iterator to the face term error
        for(unsigned int face_no=0;face_no<GeometryInfo<dim>::faces_per_cell;++face_no)
           face_integrals[cell->face(face_no)]=-1e20;
        Assert(face_integrals.size()<5,ExcInternalError());
        
        //compute the face_term, stored in face_integrals, and add them to error_indicators
        for(unsigned int face_no=0;face_no<GeometryInfo<dim>::faces_per_cell;++face_no){
           integrate_over_face(cell,face_no,face_data,primal_solution,dual_weights,face_integrals);
           error_indicators(present_cell) += face_integrals[cell->face(face_no)];
           }
        //std::cout<<"total_term_error on current cell is "
          //       <<*(std_cxx11::get<1>(*cell_and_error))
            //     <<std::endl;
        }
     std::cout<<"cells= "
              <<n_active_cells()
              <<", estimated error="
              <<std::accumulate(error_indicators.begin(),error_indicators.end(),0.)
              <<std::endl;
   }
   template<int dim>
   void
   WeightedResidual<dim>::
   integrate_over_cell(const SynchronousIterators<std_cxx11::tuple<
                       active_cell_iterator,Vector<float>::iterator>>&  cell_and_error,
                       CellData&                                        cell_data,   
                       const Vector<double>&                            primal_solution,
                       const Vector<double>&                            dual_weights)const{
      //compute the cell_term error
      //first, reinitialize the gradients, Jacobi determinants, etc for the given cell of type "iterator into a DoFHandler object", and the finite element associated with this object(here is the dual fe)
      //then, get the rhs_values on all the quadrature points and store them in the local vector "cell_data.rhs_values".
      //then, get the function_gradients of the primal_solution on the quadrature points and store them in the local vector "cell_data.cell_grads"
      //then, get the function_values of the dual_weights on the quadrature points of current cell and store them in the local vector "cell_data.dual_weights"
      //finally, compute cell term error: (-grad(uh)*beta, z-zh), store them in Vector<float>
      cell_data.fe_values.reinit(std_cxx11::get<0>(*cell_and_error));   
      cell_data.right_hand_side->value_list(cell_data.fe_values.get_quadrature_points(),
                                            cell_data.rhs_values); //not used here(rhs==0)
      cell_data.fe_values.get_function_gradients(primal_solution,
                                                 cell_data.cell_grads);
      cell_data.fe_values.get_function_values(dual_weights,
                                              cell_data.dual_weights);
      double sum = 0;
      Point<dim> beta;   //a local variable, varies from quadrature_point to quadrature_point
      for(unsigned int p=0;p<cell_data.fe_values.n_quadrature_points;++p){
         if(cell_data.fe_values.quadrature_point(p)(0)<1){
             beta(0) = cell_data.fe_values.quadrature_point(p)(1); 
             beta(1) = 1.-cell_data.fe_values.quadrature_point(p)(0);
         }
         else{
             beta(0) = 2.-cell_data.fe_values.quadrature_point(p)(1);
             beta(1) = cell_data.fe_values.quadrature_point(p)(0)-1.;
         } 
         beta /= beta.norm();
         sum += ((-cell_data.cell_grads[p])*beta*cell_data.dual_weights[p]*cell_data.fe_values.JxW(p));
      }
      *(std_cxx11::get<1>(*cell_and_error)) = sum;   
   }
   template<int dim>
   void
   WeightedResidual<dim>::integrate_over_face(active_cell_iterator& cell,
                                              const unsigned int    face_no,
                                              FaceData&             face_data,
                                              Vector<double>&       primal_solution,
                                              Vector<double>&       dual_weights,
                                              FaceIntegrals&        face_integrals)const{
      const unsigned int n_q_points = face_data.fe_face_values_cell.n_quadrature_points;
      //check if the face is a boundary face, if so, compute the face_residual on it.
      if(cell->face(face_no)->at_boundary()){
         //reinit the values of current face and get the needed information 
         face_data.fe_face_values_cell.reinit(cell,face_no);
         face_data.fe_face_values_cell.get_function_values(primal_solution, face_data.face_values);

         //compute the face_residual
         Point<dim> beta;   //a local variable, varies from quadrature_point to quadrature_point
         std::vector<double> g(n_q_points);
         PrimalSolver<dim>::boundary_function.value_list
                            (face_data.fe_face_values_cell.get_quadrature_points(),g); 
         double numerical_flux = 0; //a local variable
         for(unsigned int p=0;p<n_q_points;++p){
            if(face_data.fe_face_values_cell.quadrature_point(p)(0)<1){
                beta(0) = face_data.fe_face_values_cell.quadrature_point(p)(1); 
                beta(1) = 1.-face_data.fe_face_values_cell.quadrature_point(p)(0);
            }
            else{
                beta(0) = 2.-face_data.fe_face_values_cell.quadrature_point(p)(1);
                beta(1) = face_data.fe_face_values_cell.quadrature_point(p)(0)-1.;
            } 
            beta /= beta.norm();
            const double beta_n = beta * face_data.fe_face_values_cell.normal_vector(p); 
            //compute the numerical flux on current quadrature_point
            if(beta_n>0)
               numerical_flux = beta_n * face_data.face_values[p];
            else
               numerical_flux = beta_n * g[p];
            //compute the face_residual on current quadrature_point
            face_data.face_residual[p]=face_data.face_values[p]*beta_n-numerical_flux;
         }
         //get the dual_weights
         face_data.fe_face_values_cell.get_function_values(dual_weights,face_data.dual_weights);

         //compute the face_term error
         double face_integral = 0;
         for(unsigned int p=0;p<n_q_points;++p)
            face_integral += (face_data.face_residual[p]*
                              face_data.dual_weights[p]*
                              face_data.fe_face_values_cell.JxW(p));
         face_integrals[cell->face(face_no)] = face_integral;  //remember that face_integrals is a map
         if(face_integrals[cell->face(face_no)]<-10)
            std::cout<<"error face_term !, at boundary cell"<<std::endl;
      }//if end here

      //if the neighbor is finer(i.e. the current face has children)
      else if(cell->face(face_no)->has_children()){
        const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
        const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
        //since the map is initialized with values -1e20, here need to return them to 0 firstly.
        face_integrals[cell->face(face_no)] = 0;
        //we need to cut the computation of face_integral to subfaces.  
        for(unsigned int subface_no=0;subface_no<face->n_children();++subface_no){
           //reinit the values of current subface. now the face_data.face_values only contains information of current subface
           face_data.fe_subface_values_cell.reinit(cell,face_no,subface_no);
           face_data.fe_subface_values_cell.get_function_values(primal_solution,face_data.face_values);
           //reinit the values of neighbor child's face
           const active_cell_iterator neighbor_child 
                                      = cell->neighbor_child_on_subface(face_no,subface_no);
           Assert(neighbor_child->face(neighbor_neighbor)==cell->face(face_no)->child(subface_no),
                                       ExcInternalError());
           face_data.fe_face_values_neighbor.reinit(neighbor_child,neighbor_neighbor);
           face_data.fe_face_values_neighbor.get_function_values(primal_solution,
                                                                 face_data.neighbor_values);
           double subface_numerical_flux = 0; //a local variable
           Point<dim> beta;   //a local variable, varies from quadrature_point to quadrature_point
           //compute the subface_residual for current subface
           for(unsigned int p=0;p<n_q_points;++p){
              if(face_data.fe_subface_values_cell.quadrature_point(p)(0)<1){
                beta(0) = face_data.fe_subface_values_cell.quadrature_point(p)(1); 
                beta(1) = 1.-face_data.fe_subface_values_cell.quadrature_point(p)(0);
              }
              else{
                beta(0) = 2.-face_data.fe_subface_values_cell.quadrature_point(p)(1);
                beta(1) = face_data.fe_subface_values_cell.quadrature_point(p)(0)-1.;
              } 
              beta /= beta.norm();
              const double beta_n = beta * face_data.fe_subface_values_cell.normal_vector(p); 
              //compute the numerical flux on current quadrature_point of current subface
              if(beta_n>0)
                subface_numerical_flux = beta_n * face_data.face_values[p];
              else
                subface_numerical_flux = beta_n * face_data.neighbor_values[p];
              //face_residual on current quadrature_point of current subface
              face_data.face_residual[p] = face_data.face_values[p]*beta_n-subface_numerical_flux;
           } //current subface_residual done
      
           //compute current subface_error_term(integral)
           face_data.fe_subface_values_cell.get_function_values(dual_weights,face_data.dual_weights);
           double face_integral = 0;
           for(unsigned int p=0;p<n_q_points;++p)
              face_integral += (face_data.face_residual[p]*
                                face_data.dual_weights[p]*
                                face_data.fe_subface_values_cell.JxW(p)); //subface_integral done
           face_integrals[cell->face(face_no)] += face_integral;
        }  //face_integral done
        if(face_integrals[cell->face(face_no)]<-10)
            std::cout<<"error face_term !, has children cell"<<std::endl;
      } //else if end here     

      //if the neighbor is as fine as our cell
      else if(!cell->neighbor_is_coarser(face_no)){
         //reinit the values of current face and get the needed information 
         face_data.fe_face_values_cell.reinit(cell,face_no);
         face_data.fe_face_values_cell.get_function_values(primal_solution, face_data.face_values);

         //check if the neighbor exist
         Assert(cell->neighbor(face_no).state() == IteratorState::valid, ExcInternalError());
         //reinit the values of neighbor face and store them in face_data.neighbor_values
         const unsigned int neighbor_neighbor = cell->neighbor_of_neighbor(face_no);
         const active_cell_iterator neighbor = cell->neighbor(face_no);
         face_data.fe_face_values_neighbor.reinit(neighbor,neighbor_neighbor);
         face_data.fe_face_values_neighbor.get_function_values(primal_solution,
                                                               face_data.neighbor_values);
         //compute the face_residual
         Point<dim> beta;   //a local variable, varies from quadrature_point to quadrature_point
         double numerical_flux = 0; //a local variable
         for(unsigned int p=0;p<n_q_points;++p){
            if(face_data.fe_face_values_cell.quadrature_point(p)(0)<1){
                beta(0) = face_data.fe_face_values_cell.quadrature_point(p)(1); 
                beta(1) = 1.-face_data.fe_face_values_cell.quadrature_point(p)(0);
            }
            else{
                beta(0) = 2.-face_data.fe_face_values_cell.quadrature_point(p)(1);
                beta(1) = face_data.fe_face_values_cell.quadrature_point(p)(0)-1.;
            } 
            beta /= beta.norm();
            const double beta_n = beta * face_data.fe_face_values_cell.normal_vector(p); 
            //compute the numerical flux on current quadrature_point
            if(beta_n>0)
               numerical_flux = beta_n * face_data.face_values[p];
            else
               numerical_flux = beta_n * face_data.neighbor_values[p];
            //face_residual on current quadrature_point
            face_data.face_residual[p] = face_data.face_values[p]*beta_n-numerical_flux; 
         }
         //get the dual_weights
         face_data.fe_face_values_cell.get_function_values(dual_weights,face_data.dual_weights);

         //compute the face_term error
         double face_integral = 0;
         for(unsigned int p=0;p<n_q_points;++p)
            face_integral += (face_data.face_residual[p]*
                              face_data.dual_weights[p]*
                              face_data.fe_face_values_cell.JxW(p));
         face_integrals[cell->face(face_no)] = face_integral;  //remember that face_integrals is a map
         if(face_integrals[cell->face(face_no)]<-10)
            std::cout<<"error face_term !, at normal cell"<<std::endl;
      }//else if end here

      //if the neighbor is coarser
      else{
         //reinit the values of current face and get the needed information 
         
         face_data.fe_face_values_cell.reinit(cell,face_no);
         face_data.fe_face_values_cell.get_function_values(primal_solution, face_data.face_values);

         //check if the neighbor exist and is coarser
         Assert(cell->neighbor(face_no).state() == IteratorState::valid, ExcInternalError());
         Assert(cell->neighbor_is_coarser(face_no),ExcInternalError());
         //reinit the values of neighbor face and store them in face_data.neighbor_values
         const unsigned int neighbor_neighbor = cell->neighbor_of_coarser_neighbor(face_no).first;
         const unsigned int subface_neighbor = cell->neighbor_of_coarser_neighbor(face_no).second;
         const active_cell_iterator neighbor = cell->neighbor(face_no);
         face_data.fe_subface_values_neighbor.reinit(neighbor,neighbor_neighbor,subface_neighbor);
         face_data.fe_subface_values_neighbor.get_function_values(primal_solution,
                                                               face_data.neighbor_values);
         //compute the face_residual
         Point<dim> beta;   //a local variable, varies from quadrature_point to quadrature_point
         double numerical_flux = 0; //a local variable
         for(unsigned int p=0;p<n_q_points;++p){
            if(face_data.fe_face_values_cell.quadrature_point(p)(0)<1){
                beta(0) = face_data.fe_face_values_cell.quadrature_point(p)(1); 
                beta(1) = 1.-face_data.fe_face_values_cell.quadrature_point(p)(0);
            }
            else{
                beta(0) = 2.-face_data.fe_face_values_cell.quadrature_point(p)(1);
                beta(1) = face_data.fe_face_values_cell.quadrature_point(p)(0)-1.;
            } 
            beta /= beta.norm();
            const double beta_n = beta * face_data.fe_face_values_cell.normal_vector(p); 
            //compute the numerical flux on current quadrature_point
            if(beta_n>0)
               numerical_flux = beta_n * face_data.face_values[p];
            else
               numerical_flux = beta_n * face_data.neighbor_values[p];
            //face_residual on current quadrature_point
            face_data.face_residual[p] = face_data.face_values[p]*beta_n-numerical_flux; 
         }
         //get the dual_weights
         face_data.fe_face_values_cell.get_function_values(dual_weights,face_data.dual_weights);

         //compute the face_term error
         double face_integral = 0;
         for(unsigned int p=0;p<n_q_points;++p)
            face_integral += (face_data.face_residual[p]*
                              face_data.dual_weights[p]*
                              face_data.fe_face_values_cell.JxW(p));
         face_integrals[cell->face(face_no)] = face_integral;  //remember that face_integrals is a map
         if(face_integrals[cell->face(face_no)]<-10)
            std::cout<<"error face_term !, neighbor is coarser"<<std::endl;
      }     
   }//member function end here
}

    

//template functions, was not used in current program
  template <int dim>
  void initialize_problem (Triangulation<dim>& triangulation) 
  {
    const Point<dim> bottom_left = Point<dim>();
    const Point<dim> upper_right = Point<dim>(2.,1.);   //used to specify the domain

    std::vector<unsigned int> repetitions;   //used to subdivide the original domain
    repetitions.push_back (8);
    if (dim>=2)
       repetitions.push_back (4);
    //repetitions is a vector with 2 elements of value 8 and 4.
    
    GridGenerator::subdivided_hyper_rectangle (triangulation,repetitions,bottom_left,upper_right);  //creat a rectangle triangulation, p1/p2 specify the domain
  } 
   

  template<int dim>
  void run_simulation(){
/*
      Triangulation<dim> triangulation;
      initialize_problem(triangulation);    //generate mesh
      FE_DGQ<dim>               fe;
      BoundaryValues<dim>       boundary_function;      
      PrimalSolver<dim>  dg_method(triangulation, fe, boundary_function, rhs_function); 
*/
      unsigned int primal_fe_degree = 1;
      unsigned int dual_fe_degree = 2;
      const FE_DGQ<dim>           primal_fe(primal_fe_degree);   //??may cause problem
      const FE_DGQ<dim>           dual_fe(dual_fe_degree);
      const QGauss<dim>           quadrature(dual_fe_degree+1); 
      const QGauss<dim-1>           face_quadrature(dual_fe_degree+1); 
      AdvectionSolver::WeightedResidual<dim>  dg_method(primal_fe,dual_fe,quadrature,face_quadrature);
            

      for (unsigned int cycle=0; cycle<10; ++cycle)
      {
        deallog << "Cycle " << cycle << std::endl;
        if (cycle==0)
        {
           dg_method.initialize_problem();
        }
        else
        {
           dg_method.refine_grid (); 
        }      
 
        deallog << "Number of active cells:       "
                << dg_method.n_active_cells()
                << std::endl;

        dg_method.solve_problem();
        
        deallog << "Number of degrees of freedom: "
                << dg_method.n_dofs()
                << std::endl;
        std::cout << "The functional J: "
                  <<dg_method.return_functional()
                  <<std::endl;

        dg_method.output_results (cycle);
      }

  }
}  //end of namespace Step12

int main ()
{
  try
    {
       //Step12::AdvectionSolver::PrimalSolver<2>  dg_method;
      Step12::run_simulation<2>();         
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };

  return 0;
}
   











/*
  template <int dim>
  class AdvectionProblem
  {
  public:
    AdvectionProblem ();
    void run ();

  private:
    void initialize_problem ();
    void setup_system ();
    void assemble_system ();
    void solve (Vector<double> &solution);
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    // Furthermore we want to use DG elements of degree 1 (but this is only
    // specified in the constructor). If you want to use a DG method of a
    // different degree the whole program stays the same, only replace 1 in
    // the constructor by the desired polynomial degree.
    FE_DGQ<dim>          fe;
    DoFHandler<dim>      dof_handler;

    // The next four members represent the linear system to be
    // solved. <code>system_matrix</code> and <code>right_hand_side</code> are
    // generated by <code>assemble_system()</code>, the <code>solution</code>
    // is computed in <code>solve()</code>. The <code>sparsity_pattern</code>
    // is used to determine the location of nonzero elements in
    // <code>system_matrix</code>.
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       right_hand_side;

    typedef MeshWorker::DoFInfo<dim> DoFInfo;
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;
  };

*/


// The following <code>main</code> function is similar to previous examples as
// well, and need not be commented on.






