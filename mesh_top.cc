/* Copyright 2017 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sys/unistd.h>
#include <sys/time.h>

#include "legion.h"
using namespace Legion;
/*
 * In this section we use a sequential
 * implementation of daxpy to show how
 * to create physical instances of logical
 * regions.  In later sections we will
 * show how to extend this daxpy example
 * so that it will run with sub-tasks
 * and also run in parallel.
 */
enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  TEST_TASK_ID,
  TEST_RANGE_TASK_ID,
};
enum FieldIDs {
  FID_CELL_ID,
  FID_CELL_PARTITION_COLOR,
  FID_CELL_CELL_NRANGE,
  FID_CELL_VERTEX_NRANGE,
  FID_CELL_TO_CELL_ID,
  FID_CELL_TO_CELL_PTR,
  FID_CELL_TO_VERTEX_ID,
  FID_CELL_TO_VERTEX_PTR,
  FID_VERTEX_ID,
  FID_VERTEX_PARTITION_COLOR,
};
typedef FieldAccessor<READ_WRITE,int,1,coord_t,Realm::AffineAccessor<int,1,coord_t> > AccessorRWint;
typedef FieldAccessor<READ_WRITE,Point<1>,1,coord_t,Realm::AffineAccessor<Point<1>,1,coord_t> > AccessorRWpoint;
typedef FieldAccessor<READ_WRITE,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t> > AccessorRWrect;
typedef FieldAccessor<READ_ONLY,int,1,coord_t,Realm::AffineAccessor<int,1,coord_t> > AccessorROint;
typedef FieldAccessor<READ_ONLY,Point<1>,1,coord_t,Realm::AffineAccessor<Point<1>,1,coord_t> > AccessorROpoint;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t> > AccessorROrect;

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;
  
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  
  return cur_time;
} 

#define OUTPUT_DP

double ts_start_dp;
int root = 0;

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  root = 1;
  int num_rows = 4;
  int num_rows_partition = 2;
  int have_task = 0;
  int use_image = 1;
  
  // See if we have any command line arguments to parse
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-nr"))
        num_rows = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-np"))
        num_rows_partition = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-ht"))
        have_task = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-ui"))
        use_image = atoi(command_args.argv[++i]);
    }
  }
  int num_cells = num_rows * num_rows;
  int num_vertices = (num_rows+1) * (num_rows+1);
  int num_partitions = num_rows_partition * num_rows_partition; 
  printf("Running mesh for %d cells, %d rows, %d columns, %d partitions, have taske %d, use image %d...\n", 
					num_cells, num_rows, num_rows, num_partitions, have_task, use_image);
  // index space
  IndexSpace cells_index_space = runtime->create_index_space(ctx, Rect<1>(0, num_cells-1));
  runtime->attach_name(cells_index_space, "cells_index_space");
  IndexSpace cells_to_cells_index_space = runtime->create_index_space(ctx, Rect<1>(0, num_cells*8-1));
  runtime->attach_name(cells_to_cells_index_space, "cells_to_cells_index_space");
  
  IndexSpace vertices_index_space = runtime->create_index_space(ctx, Rect<1>(0, num_vertices-1));
  runtime->attach_name(vertices_index_space, "cells_index_space");
  IndexSpace cells_to_vertices_index_space = runtime->create_index_space(ctx, Rect<1>(0, num_cells*4-1));
  runtime->attach_name(cells_to_vertices_index_space, "cells_to_vertices_index_space");
  
  // field space
  FieldSpace cells_field_space = runtime->create_field_space(ctx);
  FieldAllocator cells_allocator = runtime->create_field_allocator(ctx, cells_field_space);
  cells_allocator.allocate_field(sizeof(int), FID_CELL_ID);
  runtime->attach_name(cells_field_space, FID_CELL_ID, "FID_CELL_ID");
  cells_allocator.allocate_field(sizeof(Point<1>), FID_CELL_PARTITION_COLOR);
  runtime->attach_name(cells_field_space, FID_CELL_PARTITION_COLOR, "FID_CELL_PARTITION_COLOR");
  cells_allocator.allocate_field(sizeof(Rect<1>), FID_CELL_CELL_NRANGE);
  runtime->attach_name(cells_field_space, FID_CELL_CELL_NRANGE, "FID_CELL_CELL_NRANGE");
  cells_allocator.allocate_field(sizeof(Rect<1>), FID_CELL_VERTEX_NRANGE);
  runtime->attach_name(cells_field_space, FID_CELL_VERTEX_NRANGE, "FID_CELL_VERTEX_NRANGE");
  
  FieldSpace cells_to_cells_field_space = runtime->create_field_space(ctx);
  FieldAllocator cells_to_cells_allocator = runtime->create_field_allocator(ctx, cells_to_cells_field_space);
  cells_to_cells_allocator.allocate_field(sizeof(int), FID_CELL_TO_CELL_ID);
  runtime->attach_name(cells_to_cells_field_space, FID_CELL_TO_CELL_ID, "FID_CELL_TO_CELL_ID");
  cells_to_cells_allocator.allocate_field(sizeof(Point<1>), FID_CELL_TO_CELL_PTR);
  runtime->attach_name(cells_to_cells_field_space, FID_CELL_TO_CELL_PTR, "FID_CELL_TO_CELL_PTR");
  
  FieldSpace cells_to_vertices_field_space = runtime->create_field_space(ctx);
  FieldAllocator cells_to_vertices_allocator = runtime->create_field_allocator(ctx, cells_to_vertices_field_space);
  cells_to_vertices_allocator.allocate_field(sizeof(int), FID_CELL_TO_VERTEX_ID);
  runtime->attach_name(cells_to_vertices_field_space, FID_CELL_TO_VERTEX_ID, "FID_CELL_TO_VERTEX_ID");
  cells_to_vertices_allocator.allocate_field(sizeof(Point<1>), FID_CELL_TO_VERTEX_PTR);
  runtime->attach_name(cells_to_vertices_field_space, FID_CELL_TO_VERTEX_PTR, "FID_CELL_TO_VERTEX_PTR");
  
  FieldSpace vertices_field_space = runtime->create_field_space(ctx);
  FieldAllocator vertices_allocator = runtime->create_field_allocator(ctx, vertices_field_space);
  vertices_allocator.allocate_field(sizeof(int), FID_VERTEX_ID);
  runtime->attach_name(vertices_field_space, FID_VERTEX_ID, "FID_VERTEX_ID");
  vertices_allocator.allocate_field(sizeof(Point<1>), FID_VERTEX_PARTITION_COLOR);
  runtime->attach_name(vertices_field_space, FID_VERTEX_PARTITION_COLOR, "FID_VERTEX_PARTITION_COLOR");
  
  // logical region
  LogicalRegion all_cells_lr = runtime->create_logical_region(ctx,cells_index_space,cells_field_space);
  runtime->attach_name(all_cells_lr, "all_cells_lr");
  LogicalRegion all_cells_to_cells_lr = runtime->create_logical_region(ctx,cells_to_cells_index_space,cells_to_cells_field_space);
  runtime->attach_name(all_cells_to_cells_lr, "all_cells_to_cells_lr");
  LogicalRegion all_cells_to_vertices_lr = runtime->create_logical_region(ctx,cells_to_vertices_index_space,cells_to_vertices_field_space);
  runtime->attach_name(all_cells_to_vertices_lr, "all_cells_to_vertices_lr");
  LogicalRegion all_vertices_lr = runtime->create_logical_region(ctx,vertices_index_space,vertices_field_space);
  runtime->attach_name(all_vertices_lr, "all_vertices_lr");
  
  // region requirement
  RegionRequirement cells_rr(all_cells_lr, READ_WRITE, EXCLUSIVE, all_cells_lr);
  cells_rr.add_field(FID_CELL_ID);
  cells_rr.add_field(FID_CELL_PARTITION_COLOR);
  cells_rr.add_field(FID_CELL_CELL_NRANGE);
  cells_rr.add_field(FID_CELL_VERTEX_NRANGE);
  
  RegionRequirement cells_to_cells_rr(all_cells_to_cells_lr, READ_WRITE, EXCLUSIVE, all_cells_to_cells_lr);
  cells_to_cells_rr.add_field(FID_CELL_TO_CELL_ID);
  cells_to_cells_rr.add_field(FID_CELL_TO_CELL_PTR);
  
  RegionRequirement cells_to_vertices_rr(all_cells_to_vertices_lr, READ_WRITE, EXCLUSIVE, all_cells_to_vertices_lr);
  cells_to_vertices_rr.add_field(FID_CELL_TO_VERTEX_ID);
  cells_to_vertices_rr.add_field(FID_CELL_TO_VERTEX_PTR);
  
  RegionRequirement vertices_rr(all_vertices_lr, READ_WRITE, EXCLUSIVE, all_vertices_lr);
  vertices_rr.add_field(FID_VERTEX_ID);
  vertices_rr.add_field(FID_VERTEX_PARTITION_COLOR);
  
  // physical region                                 
  PhysicalRegion cells_pr = runtime->map_region(ctx, cells_rr);
  cells_pr.wait_until_valid();
  
  PhysicalRegion cells_to_cells_pr = runtime->map_region(ctx, cells_to_cells_rr);
  cells_to_cells_pr.wait_until_valid();
  
  PhysicalRegion cells_to_vertices_pr = runtime->map_region(ctx, cells_to_vertices_rr);
  cells_to_vertices_pr.wait_until_valid();
  
  PhysicalRegion vertices_pr = runtime->map_region(ctx, vertices_rr);
  vertices_pr.wait_until_valid();
  
  //init the mesh
  const AccessorRWint cells_id_acc(cells_pr, FID_CELL_ID);
  const AccessorRWpoint cells_color_acc(cells_pr, FID_CELL_PARTITION_COLOR);
  const AccessorRWrect cells_cells_nrange_acc(cells_pr, FID_CELL_CELL_NRANGE);
  const AccessorRWrect cells_vertices_nrange_acc(cells_pr, FID_CELL_VERTEX_NRANGE);
  
  const AccessorRWint cells_to_cells_id_acc(cells_to_cells_pr, FID_CELL_TO_CELL_ID);
  const AccessorRWpoint cells_to_cells_ptr_acc(cells_to_cells_pr, FID_CELL_TO_CELL_PTR);
  
  const AccessorRWint cells_to_vertices_id_acc(cells_to_vertices_pr, FID_CELL_TO_VERTEX_ID);
  const AccessorRWpoint cells_to_vertices_ptr_acc(cells_to_vertices_pr, FID_CELL_TO_VERTEX_PTR);
  
  const AccessorRWint vertices_id_acc(vertices_pr, FID_VERTEX_ID);
  const AccessorRWpoint vertices_color_acc(vertices_pr, FID_VERTEX_PARTITION_COLOR);
  
  int i, j; 
  int x_idx, y_idx, x_par_idx, y_par_idx, nb_rows_per_partition;
  nb_rows_per_partition = num_rows / num_rows_partition;
  for (i = 0; i < num_cells; i++) {
    cells_id_acc[i] = i;
    cells_cells_nrange_acc[i] = Rect<1>(i*8, i*8+7);
    cells_vertices_nrange_acc[i] = Rect<1>(i*4, i*4+3);
    
    y_idx = i / num_rows;
    x_idx = i % num_rows;
    y_par_idx = y_idx / nb_rows_per_partition;
    x_par_idx = x_idx / nb_rows_per_partition;
    cells_color_acc[i] = y_par_idx * num_rows_partition + x_par_idx;
    
    cells_to_cells_id_acc[i*8+0] = i-1;
    cells_to_cells_id_acc[i*8+1] = i+1;
    cells_to_cells_id_acc[i*8+2] = i-num_rows;
    cells_to_cells_id_acc[i*8+3] = i+num_rows;
    cells_to_cells_id_acc[i*8+4] = i-num_rows-1;
    cells_to_cells_id_acc[i*8+5] = i-num_rows+1;
    cells_to_cells_id_acc[i*8+6] = i+num_rows-1;
    cells_to_cells_id_acc[i*8+7] = i+num_rows+1;
    
    // top
    if (i < num_rows) {
     // cells_to_cells_id_acc[i*4+2] = i + num_rows * (num_rows - 1);
      cells_to_cells_id_acc[i*8+2] = cells_to_cells_id_acc[i*8+3];
      cells_to_cells_id_acc[i*8+4] = cells_to_cells_id_acc[i*8+3];
      cells_to_cells_id_acc[i*8+5] = cells_to_cells_id_acc[i*8+3];
    }
    
    // bottom
    if (i >= num_rows * (num_rows - 1)) {
    //  cells_to_cells_id_acc[i*4+3] = i - num_rows * (num_rows - 1);
      cells_to_cells_id_acc[i*8+3] = cells_to_cells_id_acc[i*8+2];
      cells_to_cells_id_acc[i*8+6] = cells_to_cells_id_acc[i*8+2];
      cells_to_cells_id_acc[i*8+7] = cells_to_cells_id_acc[i*8+2];
    }
    
    // left
    if (i % num_rows == 0) {
     // cells_to_cells_id_acc[i*4+0] = i + num_rows - 1;
      cells_to_cells_id_acc[i*8+0] = cells_to_cells_id_acc[i*8+1];
      cells_to_cells_id_acc[i*8+4] = cells_to_cells_id_acc[i*8+1];
      cells_to_cells_id_acc[i*8+6] = cells_to_cells_id_acc[i*8+1];
    }
    
    // right
    if (i % num_rows == num_rows - 1) {
      //cells_to_cells_id_acc[i*4+1] = i - (num_rows - 1);
      cells_to_cells_id_acc[i*8+1] = cells_to_cells_id_acc[i*8+0];
      cells_to_cells_id_acc[i*8+5] = cells_to_cells_id_acc[i*8+0];
      cells_to_cells_id_acc[i*8+7] = cells_to_cells_id_acc[i*8+0];
    }
    
    cells_to_cells_ptr_acc[i*8+0] = Point<1>(cells_to_cells_id_acc[i*8+0]);
    cells_to_cells_ptr_acc[i*8+1] = Point<1>(cells_to_cells_id_acc[i*8+1]);
    cells_to_cells_ptr_acc[i*8+2] = Point<1>(cells_to_cells_id_acc[i*8+2]);
    cells_to_cells_ptr_acc[i*8+3] = Point<1>(cells_to_cells_id_acc[i*8+3]);
    cells_to_cells_ptr_acc[i*8+4] = Point<1>(cells_to_cells_id_acc[i*8+4]);
    cells_to_cells_ptr_acc[i*8+5] = Point<1>(cells_to_cells_id_acc[i*8+5]);
    cells_to_cells_ptr_acc[i*8+6] = Point<1>(cells_to_cells_id_acc[i*8+6]);
    cells_to_cells_ptr_acc[i*8+7] = Point<1>(cells_to_cells_id_acc[i*8+7]);
    
    cells_to_vertices_id_acc[i*4+0] = i + y_idx;
    cells_to_vertices_id_acc[i*4+1] = cells_to_vertices_id_acc[i*4+0] + 1;
    cells_to_vertices_id_acc[i*4+2] = i + y_idx + (num_rows+1);
    cells_to_vertices_id_acc[i*4+3] = cells_to_vertices_id_acc[i*4+2] + 1;
    
    cells_to_vertices_ptr_acc[i*4+0] = Point<1>(cells_to_vertices_id_acc[i*4+0]);
    cells_to_vertices_ptr_acc[i*4+1] = Point<1>(cells_to_vertices_id_acc[i*4+1]);
    cells_to_vertices_ptr_acc[i*4+2] = Point<1>(cells_to_vertices_id_acc[i*4+2]);
    cells_to_vertices_ptr_acc[i*4+3] = Point<1>(cells_to_vertices_id_acc[i*4+3]);
  //  printf("cell %d, color %lld, left %d, right %d, top %d, bottom %d, nrange(%d, %d), cell2cell neighbors: ", cells_id_acc[i], cells_color_acc[i].x, cells_to_cells_id_acc[i*8+0], cells_to_cells_id_acc[i*8+1],
    //       cells_to_cells_id_acc[i*8+2], cells_to_cells_id_acc[i*8+3], i*8, i*4+7);
    for (j = i*8; j <= i*8+7; j++) {
    //  printf("%d ", cells_to_cells_id_acc[j]);
    } 
   // printf(" cell2vertex: ");
    for (j = i*4; j <= i*4+3; j++) {
     // printf("%d ", cells_to_vertices_id_acc[j]);
    } 
 //   printf("\n");
  }
  printf("nb_rows_per_partition %d, num_rows_partition %d", nb_rows_per_partition, num_rows_partition);
  for (i = 0; i < num_vertices; i++) {
    vertices_id_acc[i] = i;
    y_idx = i / (num_rows+1);
    x_idx = i % (num_rows+1);
    y_par_idx = y_idx / nb_rows_per_partition;
    x_par_idx = x_idx / nb_rows_per_partition;
    if (y_par_idx >= num_rows_partition) y_par_idx = num_rows_partition - 1;
    if (x_par_idx >= num_rows_partition) x_par_idx = num_rows_partition - 1;
    vertices_color_acc[i] = y_par_idx * num_rows_partition + x_par_idx;
 //   printf("vertex %d, color %lld, y_idx %d, x_idx %d, y_par_idx %d, x_par_idx %d\n", vertices_id_acc[i], vertices_color_acc[i].x, y_idx, x_idx, y_par_idx, x_par_idx);
  }
 
  runtime->issue_execution_fence(ctx); 
  printf("\nstart DP\n");
  //Future f_start = runtime->get_current_time_in_microseconds(ctx);
  ts_start_dp = get_cur_time();  

  // DP
  IndexSpace partition_is = runtime->create_index_space(ctx, Rect<1>(0, num_partitions-1));
  IndexPartition owned_ip = runtime->create_partition_by_field(ctx, all_cells_lr,
                                                               all_cells_lr,
                                                               FID_CELL_PARTITION_COLOR,
                                                               partition_is);
  runtime->attach_name(owned_ip, "owned_ip");
  
  IndexPartition owned_image_nrange_ip = runtime->create_partition_by_image_range(ctx, all_cells_to_cells_lr.get_index_space(),
                                                                                runtime->get_logical_partition(all_cells_lr, owned_ip), 
                                                                                all_cells_lr,
                                                                                FID_CELL_CELL_NRANGE,
                                                                                partition_is);
  
  IndexPartition reachable_ip = runtime->create_partition_by_image(ctx, all_cells_lr.get_index_space(),
                                                                   runtime->get_logical_partition(all_cells_to_cells_lr, owned_image_nrange_ip), 
                                                                   all_cells_to_cells_lr,
                                                                   FID_CELL_TO_CELL_PTR,
                                                                   partition_is);
  runtime->attach_name(reachable_ip, "reachable_ip");
  
  IndexPartition ghost_ip = runtime->create_partition_by_difference(ctx, all_cells_lr.get_index_space(),
                                                           reachable_ip, owned_ip, partition_is);  
  
  IndexPartition ghost_preimage_ip;
  IndexPartition ghost_preimage_preimage_nrange_ip;
  if (!use_image) {
    ghost_preimage_ip = runtime->create_partition_by_preimage(ctx, ghost_ip , 
                                                               all_cells_to_cells_lr, all_cells_to_cells_lr,
                                                               FID_CELL_TO_CELL_PTR,
                                                               partition_is);
                                                                     
    ghost_preimage_preimage_nrange_ip = runtime->create_partition_by_preimage_range(ctx, ghost_preimage_ip , 
                                                                      all_cells_lr, all_cells_lr,
                                                                      FID_CELL_CELL_NRANGE,
                                                                      partition_is);
  } else {
                                                           
    ghost_preimage_ip = runtime->create_partition_by_image_range(ctx, all_cells_to_cells_lr.get_index_space(),
                                                                 runtime->get_logical_partition(all_cells_lr, ghost_ip), 
                                                                 all_cells_lr,
                                                                 FID_CELL_CELL_NRANGE,
                                                                 partition_is);

    ghost_preimage_preimage_nrange_ip = runtime->create_partition_by_image(ctx, all_cells_lr.get_index_space(),
                                                                    runtime->get_logical_partition(all_cells_to_cells_lr, ghost_preimage_ip), 
                                                                    all_cells_to_cells_lr,
                                                                    FID_CELL_TO_CELL_PTR,
                                                                    partition_is);      
  }                                                                                                                                                                                   
  IndexPartition shared_ip = runtime->create_partition_by_intersection(ctx, all_cells_lr.get_index_space(),
                                                           ghost_preimage_preimage_nrange_ip, owned_ip, partition_is);  
  runtime->attach_name(shared_ip, "shared_ip");
  
  IndexPartition private_ip = runtime->create_partition_by_difference(ctx, all_cells_lr.get_index_space(),
                                                           owned_ip, shared_ip, partition_is); 
  runtime->attach_name(private_ip, "private_ip");
  
  
  IndexPartition owned_vertices_ip = runtime->create_partition_by_field(ctx, all_vertices_lr,
                                                               all_vertices_lr,
                                                               FID_VERTEX_PARTITION_COLOR,
                                                               partition_is);
  runtime->attach_name(owned_vertices_ip, "owned_vertices_ip");
  
  IndexPartition ghost_c2v_image_nrange_ip = runtime->create_partition_by_image_range(ctx, all_cells_to_vertices_lr.get_index_space(),
                                                                                runtime->get_logical_partition(all_cells_lr, ghost_ip), 
                                                                                all_cells_lr,
                                                                                FID_CELL_VERTEX_NRANGE,
                                                                                partition_is);
  
  IndexPartition ghost_cells_vertices_ip = runtime->create_partition_by_image(ctx, all_vertices_lr.get_index_space(),
                                                                   runtime->get_logical_partition(all_cells_to_vertices_lr, ghost_c2v_image_nrange_ip), 
                                                                   all_cells_to_vertices_lr,
                                                                   FID_CELL_TO_VERTEX_PTR,
                                                                   partition_is);
  
  
  IndexPartition ghost_vertices_ip = runtime->create_partition_by_difference(ctx, all_vertices_lr.get_index_space(),
                                                           ghost_cells_vertices_ip, owned_vertices_ip, partition_is); 
  runtime->attach_name(ghost_vertices_ip, "ghost_vertices_ip");
  
  IndexPartition shared_c2v_image_nrange_ip = runtime->create_partition_by_image_range(ctx, all_cells_to_vertices_lr.get_index_space(),
                                                                                runtime->get_logical_partition(all_cells_lr, shared_ip), 
                                                                                all_cells_lr,
                                                                                FID_CELL_VERTEX_NRANGE,
                                                                                partition_is);
  
  IndexPartition shared_cells_vertices_ip = runtime->create_partition_by_image(ctx, all_vertices_lr.get_index_space(),
                                                                   runtime->get_logical_partition(all_cells_to_vertices_lr, shared_c2v_image_nrange_ip), 
                                                                   all_cells_to_vertices_lr,
                                                                   FID_CELL_TO_VERTEX_PTR,
                                                                   partition_is);
  
  
  IndexPartition shared_vertices_ip = runtime->create_partition_by_intersection(ctx, all_vertices_lr.get_index_space(),
                                                           shared_cells_vertices_ip, owned_vertices_ip, partition_is); 
  runtime->attach_name(shared_vertices_ip, "shared_vertices_ip");
  
  //runtime->issue_execution_fence(ctx);
  //Future f_end_1 = runtime->get_current_time_in_microseconds(ctx);
  //double ts_end_1 = f_end_1.get_result<long long>();
  //double sim_time_1 = 1e-6 * (ts_end_1 - ts_start);
  //printf("ELAPSED TIME = %7.3f s\n", sim_time_1);  

  // get lp
  LogicalPartition owned_lp = runtime->get_logical_partition(ctx, all_cells_lr, owned_ip);
  runtime->attach_name(owned_lp, "owned_lp");
  LogicalPartition reachable_lp = runtime->get_logical_partition(ctx, all_cells_lr, reachable_ip);
  runtime->attach_name(reachable_lp, "reachable_lp");
  LogicalPartition ghost_lp = runtime->get_logical_partition(ctx, all_cells_lr, ghost_ip);
  runtime->attach_name(ghost_lp, "ghost_lp");
  LogicalPartition ghost_preimage_preimage_nrange_lp = runtime->get_logical_partition(ctx, all_cells_lr, ghost_preimage_preimage_nrange_ip);
  runtime->attach_name(ghost_preimage_preimage_nrange_lp, "ghost_preimage_preimage_nrange_lp");
  LogicalPartition shared_lp = runtime->get_logical_partition(ctx, all_cells_lr, shared_ip);
  runtime->attach_name(shared_lp, "shared_lp");
  LogicalPartition private_lp = runtime->get_logical_partition(ctx, all_cells_lr, private_ip);
  runtime->attach_name(private_lp, "private_lp");
  
  LogicalPartition owned_vertices_lp = runtime->get_logical_partition(ctx, all_vertices_lr, owned_vertices_ip);
  runtime->attach_name(owned_vertices_lp, "owned_vertices_lp");
  LogicalPartition ghost_vertices_lp = runtime->get_logical_partition(ctx, all_vertices_lr, ghost_vertices_ip);
  runtime->attach_name(ghost_vertices_lp, "ghost_vertices_lp");
  LogicalPartition shared_vertices_lp = runtime->get_logical_partition(ctx, all_vertices_lr, shared_vertices_ip);
  runtime->attach_name(shared_vertices_lp, "shared_vertices_lp");
  

  runtime->issue_execution_fence(ctx);
  //Future f_end = runtime->get_current_time_in_microseconds(ctx);
  double ts_end = get_cur_time();
  double sim_time = (ts_end - ts_start_dp);
  printf("ELAPSED TIME = %7.3f s\n", sim_time);

  // verify task
	ArgumentMap arg_map;
  if (have_task) {
  IndexLauncher test_launcher(TEST_TASK_ID, partition_is, TaskArgument(NULL, 0), arg_map);
  test_launcher.add_region_requirement(
        RegionRequirement(owned_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_cells_lr));
  test_launcher.region_requirements[0].add_field(FID_CELL_ID);
  test_launcher.region_requirements[0].add_field(FID_CELL_PARTITION_COLOR);
  
  
  test_launcher.add_region_requirement(
        RegionRequirement(reachable_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_cells_lr));
  test_launcher.region_requirements[1].add_field(FID_CELL_ID);
  
  test_launcher.add_region_requirement(
        RegionRequirement(ghost_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_cells_lr));
  test_launcher.region_requirements[2].add_field(FID_CELL_ID);
	
  test_launcher.add_region_requirement(
        RegionRequirement(shared_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_cells_lr));
  test_launcher.region_requirements[3].add_field(FID_CELL_ID);
  
  test_launcher.add_region_requirement(
        RegionRequirement(private_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_cells_lr));
  test_launcher.region_requirements[4].add_field(FID_CELL_ID);
  
  test_launcher.add_region_requirement(
        RegionRequirement(owned_vertices_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_vertices_lr));
  test_launcher.region_requirements[5].add_field(FID_VERTEX_ID);
  test_launcher.region_requirements[5].add_field(FID_VERTEX_PARTITION_COLOR);
  
  test_launcher.add_region_requirement(
        RegionRequirement(ghost_vertices_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_vertices_lr));
  test_launcher.region_requirements[6].add_field(FID_VERTEX_ID);
  
  test_launcher.add_region_requirement(
        RegionRequirement(shared_vertices_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_vertices_lr));
  test_launcher.region_requirements[7].add_field(FID_VERTEX_ID);
  
  test_launcher.add_region_requirement(
        RegionRequirement(ghost_preimage_preimage_nrange_lp, 0/*projection ID*/,
                          READ_ONLY, EXCLUSIVE, all_cells_lr));
  test_launcher.region_requirements[8].add_field(FID_CELL_ID);
	
  FutureMap fm = runtime->execute_index_space(ctx, test_launcher);
  fm.wait_all_results(); 
  runtime->issue_execution_fence(ctx);
  double ts_end_task = get_cur_time();
  double sim_time = (ts_end_task - ts_start_dp);
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
  }
  

 LogicalPartition ghost_preimage_lp = runtime->get_logical_partition(ctx, all_cells_to_cells_lr, ghost_preimage_ip);
 IndexLauncher test_range_launcher(TEST_RANGE_TASK_ID, partition_is, TaskArgument(NULL, 0), arg_map);
 test_range_launcher.add_region_requirement(
       RegionRequirement(ghost_preimage_lp, 0/*projection ID*/,
                         READ_ONLY, EXCLUSIVE, all_cells_to_cells_lr));
 test_range_launcher.region_requirements[0].add_field(FID_CELL_TO_CELL_ID);
   runtime->execute_index_space(ctx, test_range_launcher);

  runtime->destroy_logical_region(ctx, all_cells_lr);
  runtime->destroy_logical_region(ctx, all_vertices_lr);
  runtime->destroy_field_space(ctx, cells_field_space);
  runtime->destroy_field_space(ctx, vertices_field_space);
  runtime->destroy_index_space(ctx, cells_index_space);
  runtime->destroy_index_space(ctx, vertices_index_space);
}
void test_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 9);
  assert(task->regions.size() == 9);
#if defined (OUTPUT_DP)  
  const int point = task->index_point.point_data[0];
  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  printf("-----------Running DP test for point %d, host %s -----------\n", point, hostname);
  // owned 
  const AccessorROint cells_id_acc(regions[0], FID_CELL_ID);
  const AccessorROpoint cells_color_acc(regions[0], FID_CELL_PARTITION_COLOR);
  printf("Owned at point %d...\n", point);
  Domain domain_owned = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_owned); pir(); pir++)
    printf("owned Partition %d, cell id %d, partition color %lld\n", point, cells_id_acc[*pir], cells_color_acc[*pir].x);
  
  //reachable
  printf("Reachable at point %d...\n", point);
  const AccessorROint cells_id_reachable_acc(regions[1], FID_CELL_ID);
  
  Domain domain_reachable = runtime->get_index_space_domain(ctx,
                  task->regions[1].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_reachable); pir(); pir++)
    printf("reachable Partition %d, cell id %d\n", point, cells_id_reachable_acc[*pir]);
  
  printf("Ghost at point %d...\n", point);
  const AccessorROint cells_id_ghost_acc(regions[2], FID_CELL_ID);
  
  Domain domain_ghost = runtime->get_index_space_domain(ctx,
                  task->regions[2].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_ghost); pir(); pir++)
    printf("ghost Partition %d, cell id %d\n", point, cells_id_ghost_acc[*pir]);
	
  printf("ghost preimage_2 at point %d...\n", point);
  const AccessorROint cells_id_ghost_preimage_acc(regions[3], FID_CELL_ID);
  
  Domain domain_ghost_preimage = runtime->get_index_space_domain(ctx,
                  task->regions[8].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_ghost_preimage); pir(); pir++)
    printf("ghost preimage*2 Partition %d, cell id %d\n", point, cells_id_ghost_preimage_acc[*pir]);
  
  printf("Shared at point %d...\n", point);
  const AccessorROint cells_id_shared_acc(regions[3], FID_CELL_ID);
  
  Domain domain_shared = runtime->get_index_space_domain(ctx,
                  task->regions[3].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_shared); pir(); pir++)
    printf("shared Partition %d, cell id %d\n", point, cells_id_shared_acc[*pir]);
  
  printf("Private at point %d...\n", point);
  const AccessorROint cells_id_private_acc(regions[4], FID_CELL_ID);
  
  Domain domain_private = runtime->get_index_space_domain(ctx,
                  task->regions[4].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_private); pir(); pir++)
    printf("private Partition %d, cell id %d\n", point, cells_id_private_acc[*pir]);
  
  printf("Owned vertices at point %d...\n", point);
  const AccessorROint vertices_id_owned_acc(regions[5], FID_VERTEX_ID);
  Domain domain_owned_vertices = runtime->get_index_space_domain(ctx,
                  task->regions[5].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_owned_vertices); pir(); pir++)
    printf("Owned vertices Partition %d, vertex id %d\n", point, vertices_id_owned_acc[*pir]);
  
  printf("Ghost vertices at point %d...\n", point);
  const AccessorROint vertices_id_ghost_acc(regions[6], FID_VERTEX_ID);
  Domain domain_ghost_vertices = runtime->get_index_space_domain(ctx,
                  task->regions[6].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_ghost_vertices); pir(); pir++)
    printf("Ghost vertices Partition %d, vertex id %d\n", point, vertices_id_ghost_acc[*pir]);
  
  printf("Shared vertices at point %d...\n", point);
  const AccessorROint vertices_id_shared_acc(regions[7], FID_VERTEX_ID);
  Domain domain_shared_vertices = runtime->get_index_space_domain(ctx,
                  task->regions[7].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_shared_vertices); pir(); pir++)
    printf("Shared vertices Partition %d, vertex id %d\n", point, vertices_id_shared_acc[*pir]);
#endif
}
void test_range_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  const int point = task->index_point.point_data[0];
  // owned 
  const AccessorROint cells_id_acc(regions[0], FID_CELL_TO_CELL_ID);
  printf("Check ghost preimage/image for point %d...\n", point);
  Domain domain_owned = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInDomainIterator<1> pir(domain_owned); pir(); pir++)
    printf("Partition %d, cell id %d\n", point, cells_id_acc[*pir]);
}
int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  
  {
    TaskVariantRegistrar registrar(TEST_TASK_ID, "test");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<test_task>(registrar, "test");
  }
    
  {
      TaskVariantRegistrar registrar(TEST_RANGE_TASK_ID, "test_range");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<test_range_task>(registrar, "test_range");
    }
  int rtval = Runtime::start(argc, argv);
  if (root) {
    double ts_end = get_cur_time();
    double sim_time = (ts_end - ts_start_dp);
    printf("TOTAL ELAPSED TIME = %7.3f s\n", sim_time);
  }
  return rtval;
}