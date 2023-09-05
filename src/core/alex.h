// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * ALEX with key type T and payload type P, combined type V=std::pair<T, P>.
 * Iterating through keys is done using an "Iterator".
 * Iterating through tree nodes is done using a "NodeIterator".
 *
 * Core user-facing API of Alex:
 * - Alex()
 * - void bulk_load(V values[], int num_keys)
 * - void insert(T key, P payload)
 * - Iterator begin()
 * - Iterator end()
 * - Iterator lower_bound(T key)
 * - Iterator upper_bound(T key)
 *
 * User-facing API of Iterator:
 * - void operator ++ ()  // post increment
 * - V operator * ()  // does not return reference to V by default
 * - const T& key ()
 * - P& payload ()
 * - bool is_end()
 * - bool operator == (const Iterator & rhs)
 * - bool operator != (const Iterator & rhs)
 */

#pragma once

#include <fstream>
#include <iostream>
#include <stack>
#include <type_traits>
#include <iomanip> //only for printing some debugging message.

#include "alex_base.h"
#include "alex_fanout_tree.h"
#include "alex_nodes.h"

// Whether we account for floating-point precision issues when traversing down
// ALEX.
// These issues rarely occur in practice but can cause incorrect behavior.
// Turning this on will cause slight performance overhead due to extra
// computation and possibly accessing two data nodes to perform a lookup.
#define ALEX_SAFE_LOOKUP 1

namespace alex {

template <class T, class P, class Compare = AlexCompare,
          class Alloc = std::allocator<std::pair<AlexKey<T>, P>>,
          bool allow_duplicates = true>
class Alex {
  static_assert(std::is_arithmetic<T>::value, "ALEX key type must be numeric.");
  static_assert(std::is_same<Compare, AlexCompare>::value,
                "Must use AlexCompare.");

 public:
  // Value type, returned by dereferencing an iterator
  typedef std::pair<AlexKey<T>, P> V;

  // ALEX class aliases
  typedef Alex<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef AlexNode<T, P, Alloc> node_type;
  typedef AlexModelNode<T, P, Alloc> model_node_type;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> data_node_type;

  // Forward declaration for iterators
  class Iterator;
  class ConstIterator;
  class ReverseIterator;
  class ConstReverseIterator;
  class NodeIterator;  // Iterates through all nodes with pre-order traversal

  node_type* root_node_ = nullptr;
  model_node_type* superroot_ =
      nullptr;  // phantom node that is the root's parent

  /* User-changeable parameters */
  struct Params {
    // When bulk loading, Alex can use provided knowledge of the expected
    // fraction of operations that will be inserts
    // For simplicity, operations are either point lookups ("reads") or inserts
    // ("writes)
    // i.e., 0 means we expect a read-only workload, 1 means write-only
    double expected_insert_frac = 1;
    // Maximum node size, in bytes. By default, 16MB.
    // Higher values result in better average throughput, but worse tail/max
    // insert latency
    int max_node_size = 1 << 24;
    // Approximate model computation: bulk load faster by using sampling to
    // train models
    bool approximate_model_computation = true;
    // Approximate cost computation: bulk load faster by using sampling to
    // compute cost
    bool approximate_cost_computation = false;
  };
  Params params_;

  /* Setting max node size automatically changes these parameters */
  struct DerivedParams {
    // The defaults here assume the default max node size of 16MB
    int max_fanout = 1 << 21;  // assumes 8-byte pointers
    int max_data_node_slots = (1 << 24) / sizeof(V);
  };
  DerivedParams derived_params_;

  /* Counters, useful for benchmarking and profiling */
  AtomicVal<int> num_keys = AtomicVal<int>(0);

 private:
  /* Structs used internally */
  /* Statistics related to the key domain.
   * The index can hold keys outside the domain, but lookups/inserts on those
   * keys will be inefficient.
   * If enough keys fall outside the key domain, then we expand the key domain.
   */
  struct InternalStats {
    T *key_domain_min_ = nullptr; // we need to initialize this for every initializer
    T *key_domain_max_ = nullptr; // we need to initialize this for every initializer
  };
  InternalStats istats_;

  /* Used when finding the best way to propagate up the RMI when splitting
   * upwards.
   * Cost is in terms of additional model size created through splitting
   * upwards, measured in units of pointers.
   * One instance of this struct is created for each node on the traversal path.
   * User should take into account the cost of metadata for new model nodes
   * (base_cost). */
  struct SplitDecisionCosts {
    static constexpr double base_cost =
        static_cast<double>(sizeof(model_node_type)) / sizeof(void*);
    // Additional cost due to this node if propagation stops at this node.
    // Equal to 0 if redundant slot exists, otherwise number of new pointers due
    // to node expansion.
    double stop_cost = 0;
    // Additional cost due to this node if propagation continues past this node.
    // Equal to number of new pointers due to node splitting, plus size of
    // metadata of new model node.
    double split_cost = 0;
  };

  // At least this many keys must be outside the domain before a domain
  // expansion is triggered.
  static const int kMinOutOfDomainKeys = 5;
  // After this many keys are outside the domain, a domain expansion must be
  // triggered.
  static const int kMaxOutOfDomainKeys = 1000;
  // When the number of max out-of-domain (OOD) keys is between the min and
  // max, expand the domain if the number of OOD keys is greater than the
  // expected number of OOD due to randomness by greater than the tolereance
  // factor.
  static const int kOutOfDomainToleranceFactor = 2;

  Compare key_less_ = Compare();
  Alloc allocator_ = Alloc();

  /*** Constructors and setters ***/

 public:
 /* basic initialization can handle up to 4 parameters
  * 1) max key length of each keys. default value is 1. 
  * 3) compare function used for comparing. Default is basic AlexCompare
  * 4) allocation function used for allocation. Default is basic allocator. */
  Alex() {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length_];
    istats_.key_domain_max_ = new T[max_key_length_];
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length_,
        STR_VAL_MAX);
    std::fill(istats_.key_domain_max_ + 1, istats_.key_domain_max_ + max_key_length_, 0);
    
    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    create_superroot();
  }

  Alex(const Compare& comp, const Alloc& alloc = Alloc())
      : key_less_(comp), allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length_];
    istats_.key_domain_max_ = new T[max_key_length_];
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length_,
        STR_VAL_MAX);
    std::fill(istats_.key_domain_max_ + 1, istats_.key_domain_max_ + max_key_length_, 0);

    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    create_superroot();
  }

  Alex(const Alloc& alloc) : allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length_];
    istats_.key_domain_max_ = new T[max_key_length_];
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length_,
        STR_VAL_MAX);
    std::fill(istats_.key_domain_max_ + 1, istats_.key_domain_max_ + max_key_length_, 0);

    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    create_superroot();
  }

  //NOTE : destruction should be done when multithreading
  ~Alex() {
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      delete_node(node_it.current());
    }
    delete_node(superroot_);
    delete[] istats_.key_domain_min_;
    delete[] istats_.key_domain_max_;
  }

  // Below 4 constructors initializes with range [first, last). 
  // The range does not need to be sorted. 
  // This creates a temporary copy of the data. 
  // If possible, we recommend directly using bulk_load() instead.
  // NEED FIX (max_key_length issue, not urgent
  //           possible but not implemented since it's not used yet.)
  template <class InputIterator>
  explicit Alex(InputIterator first, InputIterator last,
                const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : key_less_(comp), allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length_];
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length_,
              STR_VAL_MAX);
    istats_.key_domain_max_ = new T[max_key_length_];
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    std::fill(istats_.key_domain_max_ + 1, istats_.key_domain_max_ + max_key_length_, 0);

    std::vector<V> values;
    for (auto it = first; it != last; ++it) {
      values.push_back(*it);
    }
    std::sort(values.begin(), values.end(),
            [this](auto const& a, auto const& b) {return a.first < b.first;});
    bulk_load(values.data(), static_cast<int>(values.size()));
  }

  template <class InputIterator>
  explicit Alex(InputIterator first, InputIterator last,
                const Alloc& alloc = Alloc())
      : allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length_];
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length_,
              STR_VAL_MAX);
    istats_.key_domain_max_ = new T[max_key_length_];
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    std::fill(istats_.key_domain_max_ + 1, istats_.key_domain_max_ + max_key_length_, 0);

    std::vector<V> values;
    for (auto it = first; it != last; ++it) {
      values.push_back(*it);
    }
    std::sort(values.begin(), values.end(),
            [this](auto const& a, auto const& b) {return a.first < b.first;});
    bulk_load(values.data(), static_cast<int>(values.size()));
  }

  //IF YOUT WANT TO USE BELOW THREE FUNCTIONS IN MULTITHREAD ALEX,
  //PLEASE CHECK IF NO THREAD IS OPERAING FOR ALEX THAT'S BEING COPIED.
  explicit Alex(const self_type& other)
      : params_(other.params_),
        derived_params_(other.derived_params_),
        istats_(other.istats_),
        key_less_(other.key_less_),
        allocator_(other.allocator_) {
    istats_.key_domain_min_ = new T[max_key_length_];
    istats_.key_domain_max_ = new T[max_key_length_];
    std::copy(other.istats_.key_domain_min_, other.istats_.key_domain_min_ + max_key_length_,
        istats_.key_domain_min_);
    std::copy(other.istats_.key_domain_max_, other.istats_.key_domain_max_ + max_key_length_,
        istats_.key_domain_max_);
    superroot_ =
        static_cast<model_node_type*>(copy_tree_recursive(other.superroot_));
    root_node_ = superroot_->children_[0];
    num_keys.val_ = other.num_keys.val_;
  }

  Alex& operator=(const self_type& other) {
    if (this != &other) {
      for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
           node_it.next()) {
        delete_node(node_it.current());
      }
      delete_node(superroot_);
      delete[] istats_.key_domain_min_;
      delete[] istats_.key_domain_max_;
      params_ = other.params_;
      derived_params_ = other.derived_params_;
      istats_ = other.istats_;
      num_keys.val_ = other.num_keys.val_;
      key_less_ = other.key_less_;
      allocator_ = other.allocator_;
      istats_.key_domain_min_ = new T[max_key_length_];
      istats_.key_domain_max_ = new T[max_key_length_];
      std::copy(other.istats_.key_domain_min_, other.istats_.key_domain_min_ + max_key_length_,
          istats_.key_domain_min_);
      std::copy(other.istats_.key_domain_max_, other.istats_.key_domain_max_ + max_key_length_,
          istats_.key_domain_max_);
      superroot_ =
          static_cast<model_node_type*>(copy_tree_recursive(other.superroot_));
      root_node_ = superroot_->children_[0];
    }
    return *this;
  }

  void swap(const self_type& other) {
    std::swap(params_, other.params_);
    std::swap(derived_params_, other.derived_params_);
    std::swap(key_less_, other.key_less_);
    std::swap(allocator_, other.allocator_);
    
    auto arb_num_keys = num_keys.val_;
    num_keys.val_ = other.num_keys.val_;
    other.num_keys.val_ = num_keys.val_;

    std::swap(istats_.key_domain_min_, other.istats_.key_domain_min_);
    std::swap(istats_.key_domain_max_, other.istats_.key_domain_max_);
    std::swap(superroot_, other.superroot_);
    std::swap(root_node_, other.root_node_);
  }

 private:
  // Deep copy of tree starting at given node
  // ALEX SHOULDN'T BE WORKED BY OTHER THREADS IN THIS CASE.
  node_type* copy_tree_recursive(const node_type* node) {
    if (!node) return nullptr;
    if (node->is_leaf_) {
      return new (data_node_allocator().allocate(1))
          data_node_type(*static_cast<const data_node_type*>(node));
    } else {
      auto node_copy = new (model_node_allocator().allocate(1))
          model_node_type(*static_cast<const model_node_type*>(node));
      int cur = 0;
      while (cur < node_copy->num_children_) {
        node_type* child_node = node_copy->children_[cur];
        node_type* child_node_copy = copy_tree_recursive(child_node);
        int repeats = 1 << child_node_copy->duplication_factor_;
        for (int i = cur; i < cur + repeats; i++) {
          node_copy->children_[i] = child_node_copy;
        }
        cur += repeats;
      }
      return node_copy;
    }
  }

 public:
  // When bulk loading, Alex can use provided knowledge of the expected fraction
  // of operations that will be inserts
  // For simplicity, operations are either point lookups ("reads") or inserts
  // ("writes)
  // i.e., 0 means we expect a read-only workload, 1 means write-only
  // This is only useful if you set it before bulk loading
  void set_expected_insert_frac(double expected_insert_frac) {
    assert(expected_insert_frac >= 0 && expected_insert_frac <= 1);
    params_.expected_insert_frac = expected_insert_frac;
  }

  // Maximum node size, in bytes.
  // Higher values result in better average throughput, but worse tail/max
  // insert latency.
  void set_max_node_size(int max_node_size) {
    assert(max_node_size >= sizeof(V));
    params_.max_node_size = max_node_size;
    derived_params_.max_fanout = params_.max_node_size / sizeof(void*);
    derived_params_.max_data_node_slots = params_.max_node_size / sizeof(V);
  }

  // Bulk load faster by using sampling to train models.
  // This is only useful if you set it before bulk loading.
  void set_approximate_model_computation(bool approximate_model_computation) {
    params_.approximate_model_computation = approximate_model_computation;
  }

  // Bulk load faster by using sampling to compute cost.
  // This is only useful if you set it before bulk loading.
  void set_approximate_cost_computation(bool approximate_cost_computation) {
    params_.approximate_cost_computation = approximate_cost_computation;
  }

  /*** General helpers ***/

 public:
#if ALEX_SAFE_LOOKUP
  forceinline data_node_type* get_leaf(
    AlexKey<T> key, const uint32_t worker_id,
    int mode = 1, std::vector<TraversalNode<T, P>>* traversal_path = nullptr) {
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "traveling from superroot" << std::endl;
      alex::coutLock.unlock();
#endif
      return get_leaf_from_parent(key, worker_id, superroot_, mode, traversal_path);
  }

#endif
// Return the data node that contains the key (if it exists).
// Also optionally return the traversal path to the data node.
// traversal_path should be empty when calling this function.
// The returned traversal path begins with superroot and ends with the data
// node's parent.
// Mode 0 : It's for looking the existing key. It should check boundaries.
// Mode 1 : It's for inserting new key. It checks boundaries, but could extend it.
#if ALEX_SAFE_LOOKUP
  forceinline data_node_type* get_leaf_from_parent(
      AlexKey<T> key, const uint32_t worker_id, node_type *starting_parent,
      int mode = 1, std::vector<TraversalNode<T, P>>* traversal_path = nullptr) {
#if PROFILE
    auto get_leaf_from_parent_start_time = std::chrono::high_resolution_clock::now();
#endif
    node_type* cur = starting_parent == superroot_ ? root_node_ : starting_parent;
#if PROFILE
    if (mode == 0 && starting_parent == superroot_) {
      profileStats.get_leaf_from_get_payload_superroot_call_cnt[worker_id]++;
    }
    else if (mode == 0 && starting_parent != superroot_) {
      profileStats.get_leaf_from_get_payload_directp_call_cnt[worker_id]++;
    }
    else if (mode == 1 && starting_parent == superroot_) {
      profileStats.get_leaf_from_insert_superroot_call_cnt[worker_id]++;
    }
    else {
      profileStats.get_leaf_from_insert_directp_call_cnt[worker_id]++;
    }
#endif

    if (cur->is_leaf_) {
      //normally shouldn't happen, since normally starting node is always model node.
      return static_cast<data_node_type*>(cur);
    }

    while (true) {
      auto node = static_cast<model_node_type*>(cur);
      pthread_rwlock_rdlock(&(node->children_rw_lock_));
      node_type **cur_children = node->children_;
      int num_children = node->num_children_;
      double bucketID_prediction = node->model_.predict_double(key);
      int bucketID = static_cast<int>(bucketID_prediction);
      int dir = 0; //direction of seraching between buckets. 1 for right, -1 for left.
      bucketID =
          std::min<int>(std::max<int>(bucketID, 0), num_children - 1);
      cur = cur_children[bucketID];
      memory_fence();
      int cur_duplication_factor = 1 << cur->duplication_factor_;
      memory_fence();
      bucketID = bucketID - (bucketID % cur_duplication_factor);

#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "initial bucket : " << bucketID << std::endl;
        //std::cout << "min_key : " << cur->min_key_.read() << std::endl;
        //std::cout << "max_key : " << cur->max_key_.read() << std::endl;
        alex::coutLock.unlock();
#endif

      AlexKey<T> min_tmp_key(istats_.key_domain_min_);
      AlexKey<T> max_tmp_key(istats_.key_domain_max_);
      AlexKey<T> *cur_node_min_key = cur->min_key_.read();
      memory_fence();
      AlexKey<T> *cur_node_max_key = cur->max_key_.read();
      memory_fence();
      int was_walking_in_empty = 0;
      int smaller_than_min = key_less_(key, *(cur_node_min_key));
      int larger_than_max = key_less_(*(cur_node_max_key), key);

      if (mode == 0) {//for lookup related get_leaf
        while (smaller_than_min || larger_than_max) {
          if (smaller_than_min && larger_than_max) {
            //empty node. move according to direction.
            //could start at empty node, in this case, move left unless we're at the left end at start.
            was_walking_in_empty = 1;
            if (dir == 1 || (dir == 0 && bucketID == 0)) {
              bucketID = bucketID - (bucketID % cur_duplication_factor) + cur_duplication_factor;
              if (bucketID > num_children-1) {return nullptr;} //out of bound
              dir = 1;
            }
            else {
              bucketID = bucketID - (bucketID % cur_duplication_factor);
              if (bucketID == 0) {return nullptr;} //out of bound
              bucketID -= 1;
              dir = -1;
            }
          }
          else if (smaller_than_min) {
            bucketID = bucketID - (bucketID % cur_duplication_factor);
            if (bucketID == 0) {return nullptr;}
            if (dir == 1) {
              //it could be the case where it started from empty node, and initialized direction was wrong
              //in this case, we allow to go backward.
              if (!was_walking_in_empty) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "yo infinite loop baby!" << std::endl;
                alex::coutLock.unlock();
#endif
                return nullptr;
              }
            }
            bucketID -= 1;
            dir = -1;
            was_walking_in_empty = 0;
          }
          else if (larger_than_max) {
            bucketID = bucketID - (bucketID % cur_duplication_factor) + cur_duplication_factor;
            if (bucketID > num_children-1) {return nullptr;}
            if (dir == -1) {
              if (!was_walking_in_empty) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "yo infinite loop baby!" << std::endl;
                alex::coutLock.unlock();
#endif
              }
            }
            dir = 1;
            was_walking_in_empty = 0;
          }

          cur = cur_children[bucketID];
          memory_fence();
          cur_duplication_factor = 1 << cur->duplication_factor_;
          memory_fence();
          cur_node_min_key = cur->min_key_.read();
          memory_fence();
          cur_node_max_key = cur->max_key_.read();
          memory_fence();
          smaller_than_min = key_less_(key, *(cur_node_min_key));
          larger_than_max = key_less_(*(cur_node_max_key), key);
        }
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "decided to enter bucketID : " << bucketID << " with pointer : " << cur << '\n';
        std::cout << "bucket's min_key is " << cur_node_min_key->key_arr_
                  << " and max_key is " << cur_node_max_key->key_arr_ << std::endl;
        alex::coutLock.unlock();
#endif
      }
      else if (mode == 1) { //for insert.
        /*we need to check if inserting the key won't make collision with other node's boundary.
          If it does, we need to move to another bucket and insert it. */
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "validating insertion" << std::endl;
        alex::coutLock.unlock();
#endif
        //we first go all the way to left until min_key is smaller than our key.
        while (smaller_than_min) {
#if DEBUG_PRINT
          //alex::coutLock.lock();
          //std::cout << "t" << worker_id << " - ";
          //std::cout << "we are smaller than min (bucket ID : " << bucketID << ")" << std::endl;
          //alex::coutLock.unlock();
#endif
          bucketID = bucketID - (bucketID % cur_duplication_factor);
          if (bucketID == 0) {break;} //leftest node. we start from here.
          bucketID -= 1;
          cur = cur_children[bucketID]; 
          memory_fence();
          cur_duplication_factor = 1 << cur->duplication_factor_;
          memory_fence();
          cur_node_min_key = cur->min_key_.read();
          memory_fence();
          cur_node_max_key = cur->max_key_.read();
          memory_fence();
#if DEBUG_PRINT
            //alex::coutLock.lock();
            //std::cout << "t" << worker_id << " - ";
            //std::cout << "continuing search where min/max key is " 
            //          << cur_node_min_key->key_arr_ << " " << cur_node_max_key->key_arr_ << std::endl;
            //alex::coutLock.unlock();
#endif
          smaller_than_min = key_less_(key, *(cur_node_min_key));
          larger_than_max = key_less_(*(cur_node_max_key), key);
        }
#if DEBUG_PRINT
        //alex::coutLock.lock();
        //std::cout << "t" << worker_id << " - ";
        //std::cout << "found node whose min key is smaller than current key (bucket ID : " << bucketID << ")" << std::endl;
        //alex::coutLock.unlock();
#endif
        cur->min_key_.lock();
        memory_fence();
        cur->max_key_.lock();
        memory_fence();
        cur_node_min_key = cur->min_key_.val_;
        memory_fence();
        cur_node_max_key = cur->max_key_.val_;
        memory_fence();
#if DEBUG_PRINT
        //alex::coutLock.lock();
        //std::cout << "t" << worker_id << " - ";
        //std::cout << "node's min_key is " << cur_node_min_key->key_arr_
        //          << " and max key is " << cur_node_max_key->key_arr_ << std::endl;
        //alex::coutLock.unlock();
#endif
        smaller_than_min = key_less_(key, *(cur_node_min_key));
        larger_than_max = key_less_(*(cur_node_max_key), key);

        if (larger_than_max) { 
          //from here, we won't be reading new metadata values, so we don't rcu_progress.
          //we may read new lower level nodes or their boundaries,
          //but theoretically it won't effect the semantic
          while (true) {
            //we go on finding the next node (that should have larger keys)
            node_type *cur_next;
            int cur_bucketID = bucketID;
#if DEBUG_PRINT
            //alex::coutLock.lock();
            //std::cout << "t" << worker_id << " - ";
            //std::cout << "we are larger than max (bucket ID : " << bucketID << ")" << std::endl;
            //alex::coutLock.unlock();
#endif
            bucketID = bucketID - (bucketID % cur_duplication_factor) + cur_duplication_factor;

            if (bucketID > num_children - 1) {
              //should EXTEND the last node.
              AlexKey<T> *new_max_key = new AlexKey<T>();
              new_max_key->key_arr_ = new T[max_key_length_];
              std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
              AlexKey<T> *old_max_key = cur->max_key_.val_;
              cur->max_key_.val_ = new_max_key;
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              rcu_barrier(worker_id);
              delete old_max_key;
              bucketID = cur_bucketID;
              break;
            }

            //If we found the new node, we try to obtain the lock of those nodes.
            cur_next = cur_children[bucketID];
            cur_duplication_factor = 1 << cur_next->duplication_factor_;
#if DEBUG_PRINT
            //alex::coutLock.lock();
            //std::cout << "t" << worker_id << " - ";
            //std::cout << "found new node (bucket ID : " << bucketID << ")" << std::endl;
            //alex::coutLock.unlock();
#endif
            AlexKey<T> *next_node_min_key, *next_node_max_key;
            cur_next->min_key_.lock();
            memory_fence();
            cur_next->max_key_.lock();
            memory_fence();
            next_node_min_key = cur_next->min_key_.val_;
            memory_fence();
            next_node_max_key = cur_next->max_key_.val_;
            memory_fence();
#if DEBUG_PRINT
            //alex::coutLock.lock();
            //std::cout << "t" << worker_id << " - ";
            //std::cout << "node's min_key is " << next_node_min_key->key_arr_
            //          << " and max key is " << next_node_max_key->key_arr_ << std::endl;
            //alex::coutLock.unlock();
#endif
            smaller_than_min = key_less_(key, *(next_node_min_key));
            larger_than_max = key_less_(*(next_node_max_key), key);
              
            if (smaller_than_min && larger_than_max) {
              // next node was empty node
              // we again need to search the node after this empty node.
#if DEBUG_PRINT
              //alex::coutLock.lock();
              //std::cout << "t" << worker_id << " - ";
              //std::cout << "new node was empty node (bucket ID : " << bucketID << ")" << std::endl;
              //alex::coutLock.unlock();
#endif
EmptyNodeStart:
              node_type *cur_dbl_next;
              int cur_next_bucketID = bucketID;
              bucketID = bucketID - (bucketID % cur_duplication_factor) + cur_duplication_factor;

              if (bucketID > num_children - 1) {
                //need to insert in empty data node...
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "empty node was last node (could happen, but please verify)" << std::endl;
                alex::coutLock.unlock();
#endif

                AlexKey<T> *new_max_key = new AlexKey<T>();
                AlexKey<T> *new_min_key = new AlexKey<T>();
                new_max_key->key_arr_ = new T[max_key_length_];
                new_min_key->key_arr_ = new T[max_key_length_];
                std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
                std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_min_key->key_arr_);
                AlexKey<T> *old_max_key = cur_next->max_key_.val_;
                AlexKey<T> *old_min_key = cur_next->min_key_.val_;
                cur_next->min_key_.val_ = new_min_key;
                cur_next->max_key_.val_ = new_max_key;
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                rcu_barrier(worker_id);
                delete old_max_key;
                delete old_min_key;
                cur = cur_next;
                bucketID = cur_next_bucketID;
                break;
              }

              //If we found the new node, and if we're not in special case,
              //we try to obtain the lock of those nodes.
              cur_dbl_next = cur_children[bucketID];
              cur_duplication_factor = 1 << cur_dbl_next->duplication_factor_;
#if DEBUG_PRINT
              //alex::coutLock.lock();
              //std::cout << "t" << worker_id << " - ";
              //std::cout << "we found another node (next next node) (bucket ID : " << bucketID << ")" << std::endl;
              //alex::coutLock.unlock();
#endif
              AlexKey<T> *dbl_next_node_min_key, *dbl_next_node_max_key;
              cur_dbl_next->min_key_.lock();
              memory_fence();
              cur_dbl_next->max_key_.lock();
              memory_fence();
              dbl_next_node_min_key = cur_dbl_next->min_key_.val_;
              memory_fence();
              dbl_next_node_max_key = cur_dbl_next->max_key_.val_;
              memory_fence();
#if DEBUG_PRINT
              //alex::coutLock.lock();
              //std::cout << "t" << worker_id << " - ";
              //std::cout << "node's min_key is " << dbl_next_node_min_key->key_arr_
              //          << " and max key is " << dbl_next_node_max_key->key_arr_ << std::endl;
              //alex::coutLock.unlock();
#endif
              smaller_than_min = key_less_(key, *(dbl_next_node_min_key));
              larger_than_max = key_less_(*(dbl_next_node_max_key), key);

              if (smaller_than_min && larger_than_max) {
                //it's another empty data node
                //decided to have that new empty data node as cur_next.
                //there is no special reason of choosing that node as cur_next
                //but searching must continue becuase of possible wrong boundary.
#if DEBUG_PRINT
                //alex::coutLock.lock();
                //std::cout << "t" << worker_id << " - ";
                //std::cout << "another empty data node found. " << std::endl;
                //alex::coutLock.unlock();
#endif
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_next = cur_dbl_next;
                goto EmptyNodeStart;
              }
              else if (smaller_than_min) {
                //try inserting to empty data node. which means, we enter the empty data node!
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "decided to insert to empty data node (it's bucket ID : " << cur_next_bucketID << ")" << std::endl;
                alex::coutLock.unlock();
#endif
                AlexKey<T> *new_max_key = new AlexKey<T>();
                AlexKey<T> *new_min_key = new AlexKey<T>();
                new_max_key->key_arr_ = new T[max_key_length_];
                new_min_key->key_arr_ = new T[max_key_length_];
                std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
                std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_min_key->key_arr_);
                AlexKey<T> *old_max_key = cur_next->max_key_.val_;
                AlexKey<T> *old_min_key = cur_next->min_key_.val_;
                cur_next->min_key_.val_ = new_min_key;
                cur_next->max_key_.val_ = new_max_key;
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_dbl_next->min_key_.unlock();
                cur_dbl_next->max_key_.unlock();
                rcu_barrier(worker_id);
                delete old_max_key;
                delete old_min_key;
                cur = cur_next;
                bucketID = cur_next_bucketID;
                break;
              }
              if (larger_than_max) {
                //we are even larger than largest key of next node.
                //do the same progress as before, except that cur_dbl_next node is cur node.
#if DEBUG_PRINT
                //alex::coutLock.lock();
                //std::cout << "t" << worker_id << " - ";
                //std::cout << "we are larger than next next node (bucket ID : " << bucketID << ")" << std::endl;
                //alex::coutLock.unlock();
#endif
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_dbl_next->min_key_.unlock();
                cur_dbl_next->max_key_.unlock();
                cur = cur_dbl_next;
              }
              else {
                //the boundary may have changed to just include our key while moving
                //so choose 'cur_dbl_next' node as our moving node
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "decided to enter next next node (bucket ID : " << bucketID << ")\n";
                std::cout << "this node has min_key_ as " << cur_dbl_next->min_key_.val_->key_arr_ 
                          << " and max_key_ as " << cur_dbl_next->max_key_.val_->key_arr_ << std::endl;
                alex::coutLock.unlock();
#endif
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_dbl_next->min_key_.unlock();
                cur_dbl_next->max_key_.unlock();
                cur = cur_dbl_next;
                break;
              }
            }
            else if (smaller_than_min) {
              //Doesn't matter to enter 'cur' model node. we also extend it.
              //should do rcu barrier, since some other node may be using that boundary.
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "decided to extend current node and enter (it's bucket ID : " << cur_bucketID << ")\n";
              std::cout << "It'll have min_key_ as " << cur->min_key_.val_->key_arr_ 
                        << " and max_key_ as " << key.key_arr_ << std::endl;
              alex::coutLock.unlock();
#endif
              AlexKey<T> *new_max_key = new AlexKey<T>();
              new_max_key->key_arr_ = new T[max_key_length_];
              std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
              AlexKey<T> *old_max_key = cur->max_key_.val_;
              cur->max_key_.val_ = new_max_key;
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              cur_next->min_key_.unlock();
              cur_next->max_key_.unlock();
              rcu_barrier(worker_id);
              delete old_max_key;
              bucketID = cur_bucketID;
#if DEBUG_PRINT
              //alex::coutLock.lock();
              //std::cout << "t" << worker_id << " - ";
              //std::cout << "update finished" << std::endl;
              //alex::coutLock.unlock();
#endif
              break;
            }
            else if (larger_than_max) {
              //we are even larger than largest key of next node.
              //do the same progress as before, except that cur_next node is cur node.
#if DEBUG_PRINT
              //alex::coutLock.lock();
              //std::cout << "t" << worker_id << " - ";
              //std::cout << "we are larger than next node (bucket ID : " << bucketID << ")" << std::endl;
              //alex::coutLock.unlock();
#endif
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              cur = cur_next;
            }
            else {
              //the boundary may have changed to just include our key while moving
              //so choose 'cur_next' node as our moving node
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "decided to enter next node (it's bucket ID : " << bucketID << ")\n";
              std::cout << "this node has min_key_ as " << cur_next->min_key_.val_->key_arr_ 
                        << " and max_key_ as " << cur_next->max_key_.val_->key_arr_ << std::endl;
              alex::coutLock.unlock();
#endif
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              cur_next->min_key_.unlock();
              cur_next->max_key_.unlock();
              cur = cur_next;
              break;
            }
          }
        }
        else { //this is the next node we'll enter.
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "decided to enter current node (it's bucket ID : " << bucketID << ")\n";
          std::cout << "this node has min_key_ as " << cur->min_key_.val_->key_arr_ 
                    << " and max_key_ as " << cur->max_key_.val_->key_arr_ << std::endl;
          alex::coutLock.unlock();
#endif
          cur->min_key_.unlock();
          cur->max_key_.unlock();
        }
      }

      if (traversal_path) {
        traversal_path->push_back({node, bucketID});
      }

      pthread_rwlock_unlock(&(node->children_rw_lock_));

      if (cur->is_leaf_) {
        // we don't do rcu_progress here, since we are entering data node.
        // rcu_progress should be called at adequate point where the users finished using this data node.
        // If done ignorantly, it could cause null pointer access (because of destruction by other thread)
#if PROFILE
        auto get_leaf_from_parent_end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::fgTimeUnit>(get_leaf_from_parent_end_time - get_leaf_from_parent_start_time).count();
        if (mode == 0 && starting_parent == superroot_) {
          profileStats.get_leaf_from_get_payload_superroot_time[worker_id] += elapsed_time;
          profileStats.max_get_leaf_from_get_payload_superroot_time[worker_id] =
            std::max(profileStats.max_get_leaf_from_get_payload_superroot_time[worker_id], elapsed_time);
          profileStats.min_get_leaf_from_get_payload_superroot_time[worker_id] =
            std::min(profileStats.max_get_leaf_from_get_payload_superroot_time[worker_id], elapsed_time);
        }
        else if (mode == 0 && starting_parent != superroot_) {
          profileStats.get_leaf_from_get_payload_directp_time[worker_id] += elapsed_time;
          profileStats.max_get_leaf_from_get_payload_directp_time[worker_id] =
            std::max(profileStats.max_get_leaf_from_get_payload_directp_time[worker_id], elapsed_time);
          profileStats.min_get_leaf_from_get_payload_directp_time[worker_id] =
            std::min(profileStats.min_get_leaf_from_get_payload_directp_time[worker_id], elapsed_time);
        }
        else if (starting_parent == superroot_) {
          profileStats.get_leaf_from_insert_superroot_time[worker_id] += elapsed_time;
          profileStats.max_get_leaf_from_insert_superroot_time[worker_id] =
            std::max(profileStats.max_get_leaf_from_insert_superroot_time[worker_id], elapsed_time);
          profileStats.min_get_leaf_from_insert_superroot_time[worker_id] =
            std::min(profileStats.max_get_leaf_from_insert_superroot_time[worker_id], elapsed_time);
        }
        else {
          profileStats.get_leaf_from_insert_directp_time[worker_id] += elapsed_time;
          profileStats.max_get_leaf_from_insert_directp_time[worker_id] =
            std::max(profileStats.max_get_leaf_from_insert_directp_time[worker_id], elapsed_time);
          profileStats.min_get_leaf_from_insert_directp_time[worker_id] =
            std::min(profileStats.min_get_leaf_from_insert_directp_time[worker_id], elapsed_time);
        }
#endif
        return (data_node_type *) cur;
      }
      //entering model node, need to progress
      //chosen model nodes are never destroyed, (without erase implementation, not used currently.)
      //Synchronization issue will be checked by another while loop.
      rcu_progress(worker_id);
    }
  }
#else
  data_node_type* get_leaf(
      AlexKey<T> key, std::vector<TraversalNode>* traversal_path = nullptr) const {
    return nullptr; //not implemented
  }
#endif

 private:
  // Honestly, can't understand why below 4 functions exists 
  // (first_data_node / last_data_node / get_min_key / get_max_key)
  // (since it's declared private and not used anywhere)
  // Return left-most data node
  data_node_type* first_data_node() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_[0];
    }
    return static_cast<data_node_type*>(cur);
  }

  // Return right-most data node
  data_node_type* last_data_node() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      auto node = static_cast<model_node_type*>(cur);
      cur = node->children_[node->num_children_ - 1];
    }
    return static_cast<data_node_type*>(cur);
  }

  // Returns minimum key in the index
  T *get_min_key() const { return first_data_node()->first_key(); }

  // Returns maximum key in the index
  T *get_max_key() const { return last_data_node()->last_key(); }

  // Link all data nodes together. Used after bulk loading.
  void link_all_data_nodes() {
    data_node_type* prev_leaf = nullptr;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      node_type* cur = node_it.current();
      if (cur->is_leaf_) {
        auto node = static_cast<data_node_type*>(cur);
        if (prev_leaf != nullptr) {
          prev_leaf->next_leaf_.val_ = node;
          node->prev_leaf_.val_ = prev_leaf;
        }
        prev_leaf = node;
      }
    }
  }

  // Link the new data nodes together when old data node is replaced by two new
  // data nodes.
  void link_data_nodes(data_node_type* old_leaf,
                       data_node_type* left_leaf, data_node_type* right_leaf) {
    data_node_type *old_leaf_prev_leaf = old_leaf->prev_leaf_.read();
    data_node_type *old_leaf_next_leaf = old_leaf->next_leaf_.read();
    if (old_leaf_prev_leaf != nullptr) {
      data_node_type *olpl_pending_rl = old_leaf_prev_leaf->pending_right_leaf_.read();
      if (olpl_pending_rl != nullptr) {
        olpl_pending_rl->next_leaf_.update(left_leaf);
        left_leaf->prev_leaf_.update(olpl_pending_rl);
      }
      else {
        old_leaf_prev_leaf->next_leaf_.update(left_leaf);
        left_leaf->prev_leaf_.update(old_leaf_prev_leaf);
      }
    }
    else {
      left_leaf->prev_leaf_.update(nullptr);
    }
    left_leaf->next_leaf_.update(right_leaf);
    right_leaf->prev_leaf_.update(left_leaf);
    if (old_leaf_next_leaf != nullptr) {
      data_node_type *olnl_pending_ll = old_leaf_next_leaf->pending_left_leaf_.read();
      if (olnl_pending_ll != nullptr) {
        olnl_pending_ll->prev_leaf_.update(right_leaf);
        right_leaf->next_leaf_.update(olnl_pending_ll);
      }
      else {
        old_leaf_next_leaf->prev_leaf_.update(right_leaf);
        right_leaf->next_leaf_.update(old_leaf_next_leaf);
      }
    }
    else {
      right_leaf->next_leaf_.update(nullptr);
    }
  }

  /*** Allocators and comparators ***/

 public:
  Alloc get_allocator() const { return allocator_; }

  Compare key_comp() const { return key_less_; }

 private:
  typename model_node_type::alloc_type model_node_allocator() {
    return typename model_node_type::alloc_type(allocator_);
  }

  typename data_node_type::alloc_type data_node_allocator() {
    return typename data_node_type::alloc_type(allocator_);
  }

  typename model_node_type::pointer_alloc_type pointer_allocator() {
    return typename model_node_type::pointer_alloc_type(allocator_);
  }

  void delete_node(node_type* node) {
    if (node == nullptr) {
      return;
    } else if (node->is_leaf_) {
      data_node_allocator().destroy(static_cast<data_node_type*>(node));
      data_node_allocator().deallocate(static_cast<data_node_type*>(node), 1);
    } else {
      model_node_allocator().destroy(static_cast<model_node_type*>(node));
      model_node_allocator().deallocate(static_cast<model_node_type*>(node), 1);
    }
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Bulk loading ***/

 public:
  // values should be the sorted array of key-payload pairs.
  // The number of elements should be num_keys.
  // The index must be empty when calling this method.
  void bulk_load(const V values[], int num_keys) {
    if (this->num_keys.val_ > 0 || num_keys <= 0) {
      return;
    }
    delete_node(root_node_);  // delete the empty root node from constructor

    this->num_keys.val_ = num_keys;

    // Build temporary root model, which outputs a CDF in the range [0, 1]
    root_node_ =
        new (model_node_allocator().allocate(1)) model_node_type(0, nullptr, allocator_);
    AlexKey<T> min_key = values[0].first;
    AlexKey<T> max_key = values[num_keys - 1].first;

    if (typeid(T) == typeid(char)) { //for string key
      LinearModelBuilder<T> root_model_builder(&(root_node_->model_));
      for (int i = 0; i < num_keys; i++) {
#if DEBUG_PRINT
        //printf("adding : %f\n", (double) (i) / (num_keys-1));
#endif
        root_model_builder.add(values[i].first, (double) (i) / (num_keys-1));
      }
      root_model_builder.build();
    }
    else { //for numeric key
      std::cout << "Please use only string keys" << std::endl;
      abort();
    }
#if DEBUG_PRINT
    //for (int i = 0; i < num_keys; i++) {
    //  std::cout << "inserting " << values[i].first.key_arr_ << '\n';
    //}
    //std::cout << "left prediction result (bulk_load) " 
    //          << root_node_->model_.predict_double(values[1].first) 
    //          << std::endl;
    //std::cout << "right prediction result (bulk_load) " 
    //          << root_node_->model_.predict_double(values[num_keys-2].first) 
    //          << std::endl;
#endif

    // Compute cost of root node
    LinearModel<T> root_data_node_model;
    data_node_type::build_model(values, num_keys, &root_data_node_model,
                                params_.approximate_model_computation);
    DataNodeStats stats;
    root_node_->cost_ = data_node_type::compute_expected_cost(
        values, num_keys, data_node_type::kInitDensity_,
        params_.expected_insert_frac, &root_data_node_model,
        params_.approximate_cost_computation, &stats);

    // Recursively bulk load
    bulk_load_node(values, num_keys, root_node_, nullptr, num_keys,
                   &root_data_node_model);

    create_superroot();
    update_superroot_key_domain();
    link_all_data_nodes();

#if DEBUG_PRINT
    //std::cout << "structure's min_key after bln : " << istats_.key_domain_min_ << std::endl;
    //std::cout << "structure's max_key after bln : " << istats_.key_domain_max_ << std::endl;
#endif
  }

 private:
  // Only call this after creating a root node
  void create_superroot() {
    if (!root_node_) return;
    delete_node(superroot_);
    superroot_ = new (model_node_allocator().allocate(1))
        model_node_type(static_cast<short>(root_node_->level_ - 1), nullptr, allocator_);
    superroot_->num_children_ = 1;
    superroot_->children_ = new node_type*[1];
    root_node_->parent_ = superroot_;
    update_superroot_pointer();
  }

  // Updates the key domain based on the min/max keys and retrains the model.
  // Should only be called immediately after bulk loading
  void update_superroot_key_domain() {
    T *min_key_arr, *max_key_arr;
    //min/max should always be '!' and '~...~'
    //the reason we are doing this cumbersome process is because
    //'!' may not be inserted at the first data node.
    //We need some way to handle this. May be fixed by unbiasing keys.
    min_key_arr = (T *) malloc(max_key_length_);
    max_key_arr = (T *) malloc(max_key_length_);
    for (unsigned int i = 0; i < max_key_length_; i++) {
      max_key_arr[i] = STR_VAL_MAX;
      min_key_arr[i] = (i == 0) ? STR_VAL_MIN : 0;
    }

#if DEBUG_PRINT
    //for (unsigned int i = 0; i < max_key_length_; i++) {
    //  std::cout << min_key_arr[i] << ' ';
    //}
    //std::cout << std::endl;
    //for (unsigned int i = 0; i < max_key_length_; i++) {
    //  std::cout << max_key_arr[i] << ' ';
    //}
    //std::cout << std::endl;
#endif
    std::copy(min_key_arr, min_key_arr + max_key_length_, istats_.key_domain_min_);
    std::copy(max_key_arr, max_key_arr + max_key_length_, istats_.key_domain_max_);
    std::copy(min_key_arr, min_key_arr + max_key_length_, superroot_->min_key_.val_->key_arr_);
    std::copy(max_key_arr, max_key_arr + max_key_length_, superroot_->max_key_.val_->key_arr_);

    AlexKey<T> mintmpkey(istats_.key_domain_min_);
    AlexKey<T> maxtmpkey(istats_.key_domain_max_);
    if (key_equal(mintmpkey, maxtmpkey)) {//keys are equal
      unsigned int non_zero_cnt_ = 0;

      for (unsigned int i = 0; i < max_key_length_; i++) {
        if (istats_.key_domain_min_[i] == 0) {
          superroot_->model_.a_[i] = 0;
        }
        else {
          superroot_->model_.a_[i] = 1 / istats_.key_domain_min_[i];
          non_zero_cnt_ += 1;
        }
      }
      
      for (unsigned int i = 0; i < max_key_length_; i++) {
        superroot_->model_.a_[i] /= non_zero_cnt_;
      }
      superroot_->model_.b_ = 0;
    }
    else {//keys are not equal
      double direction_vector_[max_key_length_] = {0.0};
      
      for (unsigned int i = 0; i < max_key_length_; i++) {
        direction_vector_[i] = (double) istats_.key_domain_max_[i] - istats_.key_domain_min_[i];
      }
      superroot_->model_.b_ = 0.0;
      unsigned int non_zero_cnt_ = 0;
      for (unsigned int i = 0; i < max_key_length_; i++) {
        if (direction_vector_[i] == 0) {
          superroot_->model_.a_[i] = 0;
        }
        else {
          superroot_->model_.a_[i] = 1 / (direction_vector_[i]);
          superroot_->model_.b_ -= istats_.key_domain_min_[i] / direction_vector_[i];
          non_zero_cnt_ += 1;
        }
      }
      
      for (unsigned int i = 0; i < max_key_length_; i++) {
        superroot_->model_.a_[i] /= non_zero_cnt_;
      }
      superroot_->model_.b_ /= non_zero_cnt_;
    }

    if (typeid(T) == typeid(char)) { //need to free malloced objects.
      free(min_key_arr);
      free(max_key_arr);
    }

#if DEBUG_PRINT
    //std::cout << "left prediction result (uskd) " << superroot_->model_.predict_double(mintmpkey) << std::endl;
    //std::cout << "right prediction result (uskd) " << superroot_->model_.predict_double(maxtmpkey) << std::endl;
#endif
  }

  void update_superroot_pointer() {
    superroot_->children_[0] = root_node_;
    superroot_->level_ = static_cast<short>(root_node_->level_ - 1);
  }

  // Recursively bulk load a single node.
  // Assumes node has already been trained to output [0, 1), has cost.
  // Figures out the optimal partitioning of children.
  // node is trained as if it's a model node.
  // data_node_model is what the node's model would be if it were a data node of
  // dense keys.
  void bulk_load_node(const V values[], int num_keys, node_type*& node,
                      model_node_type* parent, int total_keys,
                      const LinearModel<T>* data_node_model = nullptr) {
    // Automatically convert to data node when it is impossible to be better
    // than current cost
#if DEBUG_PRINT
    std::cout << "called bulk_load_node!" << std::endl;
#endif
    if (num_keys <= derived_params_.max_data_node_slots *
                        data_node_type::kInitDensity_ &&
        (node->cost_ < kNodeLookupsWeight || node->model_.a_ == 0) &&
        (node != root_node_)) {
      auto data_node = new (data_node_allocator().allocate(1))
          data_node_type(node->level_, derived_params_.max_data_node_slots,
                         parent, key_less_, allocator_);
      data_node->bulk_load(values, num_keys, data_node_model,
                           params_.approximate_model_computation);
      data_node->cost_ = node->cost_;
      delete_node(node);
      node = data_node;
#if DEBUG_PRINT
      std::cout << "returned because it can't be better" << std::endl;
#endif
      return;
    }

    // Use a fanout tree to determine the best way to divide the key space into
    // child nodes
    std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;
    std::pair<int, double> best_fanout_stats;
    int max_data_node_keys = static_cast<int>(
        derived_params_.max_data_node_slots * data_node_type::kInitDensity_);
    best_fanout_stats = fanout_tree::find_best_fanout_bottom_up<T, P>(
        values, num_keys, node, total_keys, used_fanout_tree_nodes,
        derived_params_.max_fanout, max_data_node_keys,
        params_.expected_insert_frac, params_.approximate_model_computation,
        params_.approximate_cost_computation);
    int best_fanout_tree_depth = best_fanout_stats.first;
    double best_fanout_tree_cost = best_fanout_stats.second;

    // Decide whether this node should be a model node or data node
    if (best_fanout_tree_cost < node->cost_ ||
        num_keys > derived_params_.max_data_node_slots *
                       data_node_type::kInitDensity_) {
#if DEBUG_PRINT
      std::cout << "decided that current bulk_load_node calling node should be model node" << std::endl;
#endif
      // Convert to model node based on the output of the fanout tree
      auto model_node = new (model_node_allocator().allocate(1))
          model_node_type(node->level_, parent, allocator_);
      if (best_fanout_tree_depth == 0) {
        // slightly hacky: we assume this means that the node is relatively
        // uniform but we need to split in
        // order to satisfy the max node size, so we compute the fanout that
        // would satisfy that condition in expectation
        //std::cout << "hitted hacky case in bulk_load" << std::endl;
        best_fanout_tree_depth =
            std::max(static_cast<int>(std::log2(static_cast<double>(num_keys) /
                                       derived_params_.max_data_node_slots)) + 1, 1);
        //clear pointers used in fanout_tree (O(N)), and then empty used_fanout_tree_nodes.
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          delete[] tree_node.a;
        }
        used_fanout_tree_nodes.clear();
        int max_data_node_keys = static_cast<int>(
            derived_params_.max_data_node_slots * data_node_type::kInitDensity_);
#if DEBUG_PRINT
        std::cout << "computing level for depth" << std::endl;
#endif
        while (true) {
          fanout_tree::compute_level<T, P>(
            values, num_keys, total_keys, used_fanout_tree_nodes,
            best_fanout_tree_depth, max_data_node_keys,
            params_.expected_insert_frac, params_.approximate_model_computation,
            params_.approximate_cost_computation);
          
          if (used_fanout_tree_nodes.front().right_boundary == num_keys) {
            //std::cout << "retry" << '\n';
            for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
              delete[] tree_node.a;
            }
            used_fanout_tree_nodes.clear();
            best_fanout_tree_depth <<= 1;
            if (best_fanout_tree_depth > derived_params_.max_fanout) {
              std::cout << values[0].first.key_arr_ << '\n';
              std::cout << values[num_keys - 1].first.key_arr_ << '\n';
              std::cout << "bad case in bulk_load_node. unsolvable" << std::endl;
              abort();
            }
          }
          else break;
          
        }
#if DEBUG_PRINT
        std::cout << "finished level computing" << std::endl;
#endif
      }
      int fanout = 1 << best_fanout_tree_depth;
#if DEBUG_PRINT
      std::cout << "chosen fanout is... : " << fanout << std::endl;
#endif
      //obtianing CDF resulting to [0,fanout]
      LinearModel<T> tmp_model;
      LinearModelBuilder<T> tmp_model_builder(&tmp_model);
      for (int i = 0; i < num_keys; i++) {
        tmp_model_builder.add(values[i].first, ((double) i * fanout / (num_keys-1)));
      }
      tmp_model_builder.build();
      for (unsigned int i = 0; i < max_key_length_; i++) {
        model_node->model_.a_[i] = tmp_model.a_[i];
      }
      model_node->model_.b_ = tmp_model.b_; 
      
      model_node->num_children_ = fanout;
      model_node->children_ = new node_type*[fanout];

      // Instantiate all the child nodes and recurse
      int cur = 0;
#if DEBUG_PRINT
      //int cumu_repeat = 0;
#endif
      for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
        auto child_node = new (model_node_allocator().allocate(1))
            model_node_type(static_cast<short>(node->level_ + 1), model_node, allocator_);
        child_node->cost_ = tree_node.cost;
        child_node->duplication_factor_ =
            static_cast<uint8_t>(best_fanout_tree_depth - tree_node.level);
        int repeats = 1 << child_node->duplication_factor_;
#if DEBUG_PRINT
        //cumu_repeat += repeats;
        //std::cout << "started finding boundary..." << std::endl;
        //std::cout << "for left_value with : " << left_value << std::endl;
        //std::cout << "and right_value with : " << right_value << std::endl;
        //std::cout << "so covering indexes are : " << cumu_repeat - repeats << "~" << cumu_repeat - 1 << std::endl;
#endif

        //finds left/right boundary using tree_node.
#if DEBUG_PRINT
        std::cout << "finished finding boundary..." << std::endl;
        std::cout << "left boundary is : ";
        if (tree_node.left_boundary == num_keys) {
          for (unsigned int i = 0; i < max_key_length_; i++) {
            std::cout << (char) values[tree_node.left_boundary - 1].first.key_arr_[i];
          }
        }
        else {
          for (unsigned int i = 0; i < max_key_length_; i++) {
            std::cout << (char) values[tree_node.left_boundary].first.key_arr_[i];
          }
        }
        std::cout << std::endl;
        std::cout << "right boundary is : ";
        for (unsigned int i = 0; i < max_key_length_; i++) {
          std::cout << (char) values[tree_node.right_boundary - 1].first.key_arr_[i];
        }
        std::cout << std::endl;
#endif

        //obtain CDF with range [0,1]
        int num_keys = tree_node.right_boundary - tree_node.left_boundary;
        LinearModelBuilder<T> child_model_builder(&child_node->model_);
#if DEBUG_PRINT
        //printf("l_idx : %d, f_idx : %d, num_keys : %d\n", l_idx, f_idx, num_keys);
#endif
        if (num_keys == 1) {
          child_model_builder.add(values[tree_node.left_boundary].first, 1.0);
        }
        else {
          for (int i = tree_node.right_boundary; i < tree_node.left_boundary; i++) {
            child_model_builder.add(values[i].first, (double) (i-tree_node.left_boundary)/(num_keys-1));
          }
        }
        child_model_builder.build();

#if DEBUG_PRINT
        //T left_key[max_key_length_];
        //T right_key[max_key_length_];
        //for (unsigned int i = 0; i < max_key_length_; i++) {
        //  left_key[i] = left_boundary[i];
        //  right_key[i] = right_boundary[i];
        //}
        //std::cout << "left prediction result (bln) " << child_node->model_.predict_double(AlexKey<T>(left_key)) << std::endl;
        //std::cout << "right prediction result (bln) " << child_node->model_.predict_double(AlexKey<T>(right_key)) << std::endl;
#endif

        model_node->children_[cur] = child_node;
        LinearModel<T> child_data_node_model(tree_node.a, tree_node.b);
        bulk_load_node(values + tree_node.left_boundary,
                       tree_node.right_boundary - tree_node.left_boundary,
                       model_node->children_[cur], model_node, total_keys,
                       &child_data_node_model);
        model_node->children_[cur]->duplication_factor_ =
            static_cast<uint8_t>(best_fanout_tree_depth - tree_node.level);
        
        if (model_node->children_[cur]->is_leaf_) {
          static_cast<data_node_type*>(model_node->children_[cur])
              ->expected_avg_exp_search_iterations_ =
              tree_node.expected_avg_search_iterations;
          static_cast<data_node_type*>(model_node->children_[cur])
              ->expected_avg_shifts_ = tree_node.expected_avg_shifts;
        }
        for (int i = cur + 1; i < cur + repeats; i++) {
          model_node->children_[i] = model_node->children_[cur];
        }
        cur += repeats;
      }

      /* update min_key_, max_key_ for new model node*/
      std::copy(values[0].first.key_arr_, values[0].first.key_arr_ + max_key_length_,
        model_node->min_key_.val_->key_arr_);
      std::copy(values[num_keys-1].first.key_arr_, values[num_keys-1].first.key_arr_ + max_key_length_,
        model_node->max_key_.val_->key_arr_);
      
      
#if DEBUG_PRINT
      std::cout << "min_key_(model_node) : " << model_node->min_key_.val_->key_arr_ << '\n';
      std::cout << "max_key_(model_node) : " << model_node->max_key_.val_->key_arr_ << '\n';
      for (int i = 0; i < fanout; i++) {
        std::cout << i << "'s initial pointer value is : " << model_node->children_[i] << '\n';
        std::cout << i << "'s min_key is : " << model_node->children_[i]->min_key_.val_->key_arr_ << '\n';
        std::cout << i << "'s max_key is : " << model_node->children_[i]->max_key_.val_->key_arr_ << '\n';
      }
      std::cout << std::flush;
#endif

      delete_node(node);
      node = model_node;
    } else {
#if DEBUG_PRINT
      std::cout << "decided that current bulk_load_node calling node should be data node" << std::endl;
#endif
      // Convert to data node
      auto data_node = new (data_node_allocator().allocate(1))
          data_node_type(node->level_, derived_params_.max_data_node_slots,
                         parent, key_less_, allocator_);
      data_node->bulk_load(values, num_keys, data_node_model,
                           params_.approximate_model_computation);
      data_node->cost_ = node->cost_;
      delete_node(node);
      node = data_node;
    }

    //empty used_fanout_tree_nodes for preventing memory leakage.
    for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
      delete[] tree_node.a;
    }
#if DEBUG_PRINT
    std::cout << "returned using fanout" << std::endl;
#endif
  }

  // Caller needs to set the level, duplication factor, and neighbor pointers of
  // the returned data node
  static data_node_type* bulk_load_leaf_node_from_existing(
      const data_node_type* existing_node, int left, int right, uint32_t worker_id, self_type *this_ptr,
      bool compute_cost = true, const fanout_tree::FTNode* tree_node = nullptr,
      bool reuse_model = false, bool keep_left = false,
      bool keep_right = false) {
    auto node = new (this_ptr->data_node_allocator().allocate(1))
        data_node_type(existing_node->parent_, this_ptr->key_less_, this_ptr->allocator_);
    if (tree_node) {
      // Use the model and num_keys saved in the tree node so we don't have to
      // recompute it
      LinearModel<T> precomputed_model(tree_node->a, tree_node->b);
      node->bulk_load_from_existing(existing_node, left, right, worker_id, keep_left,
                                    keep_right, &precomputed_model,
                                    tree_node->num_keys);
    } else if (reuse_model) {
      // Use the model from the existing node
      // Assumes the model is accurate
      int num_actual_keys = existing_node->num_keys_in_range(left, right);
      LinearModel<T> precomputed_model(existing_node->model_);
      precomputed_model.b_ -= left;
      precomputed_model.expand(static_cast<double>(num_actual_keys) /
                               (right - left));
      node->bulk_load_from_existing(existing_node, left, right, worker_id, keep_left,
                                    keep_right, &precomputed_model,
                                    num_actual_keys);
    } else {
      node->bulk_load_from_existing(existing_node, left, right, worker_id, keep_left,
                                    keep_right);
    }
    node->max_slots_ = this_ptr->derived_params_.max_data_node_slots;
    if (compute_cost) {
      node->cost_ = node->compute_expected_cost(existing_node->frac_inserts());
    }
    return node;
  }

  /*** Lookup ***/

  size_t count(const AlexKey<T>& key) {
    ConstIterator it = lower_bound(key);
    size_t num_equal = 0;
    while (!it.is_end() && key_equal(it.key(), key)) {
      num_equal++;
      ++it;
    }
    return num_equal;
  }

  // Returns an iterator to the first key no less than the input value
  //returns end iterator on error.
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator lower_bound(const AlexKey<T>& key) {
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return end();}
    int idx = leaf->find_lower(key);
    return Iterator(leaf, idx);  // automatically handles the case where idx ==
                                 // leaf->data_capacity
  }

  typename self_type::ConstIterator lower_bound(const AlexKey<T>& key) const {
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return cend();}
    int idx = leaf->find_lower(key);
    return ConstIterator(leaf, idx);  // automatically handles the case where
                                      // idx == leaf->data_capacity
  }

  // Returns an iterator to the first key greater than the input value
  // returns end iterator on error
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator upper_bound(const AlexKey<T>& key) {
    data_node_type* leaf = typeid(T) == typeid(char) ? get_leaf(key, 0) : get_leaf(key);
    if (leaf == nullptr) {return end();}
    int idx = leaf->find_upper(key);
    return Iterator(leaf, idx);  // automatically handles the case where idx ==
                                 // leaf->data_capacity
  }

  typename self_type::ConstIterator upper_bound(const AlexKey<T>& key) const {
    data_node_type* leaf = typeid(T) == typeid(char) ? get_leaf(key, 0) : get_leaf(key);
    if (leaf == nullptr) {return cend();}
    int idx = leaf->find_upper(key);
    return ConstIterator(leaf, idx);  // automatically handles the case where
                                      // idx == leaf->data_capacity
  }

  std::pair<Iterator, Iterator> equal_range(const AlexKey<T>& key) {
    return std::pair<Iterator, Iterator>(lower_bound(key), upper_bound(key));
  }

  std::pair<ConstIterator, ConstIterator> equal_range(const AlexKey<T>& key) const {
    return std::pair<ConstIterator, ConstIterator>(lower_bound(key),
                                                   upper_bound(key));
  }

  // Returns whether payload search was successful, and the payload itself if it was successful.
  // This avoids the overhead of creating an iterator
public:
  std::tuple<int, P, model_node_type *> get_payload(const AlexKey<T>& key, int32_t worker_id) {
    return get_payload_from_parent(key, superroot_, worker_id);
  }

  //first element returns...
  //0 on success.
  //1 if failed because write is writing
  //2 if failed because not foundable
  std::tuple<int, P, model_node_type *> get_payload_from_parent(const AlexKey<T>& key, model_node_type *last_parent, int32_t worker_id) {
#if PROFILE
    if (last_parent == superroot_) {
      profileStats.get_payload_superroot_call_cnt[worker_id]++;
    }
    else {
      profileStats.get_payload_directp_call_cnt[worker_id]++;
    }
    auto get_payload_from_parent_start_time = std::chrono::high_resolution_clock::now();
#endif
    data_node_type* leaf = get_leaf_from_parent(key, worker_id, last_parent, 0);
    if (leaf == nullptr) {
      rcu_progress(worker_id);
      return {2, 0, nullptr};
    }

    //try reading. If failed, retry later
    if (pthread_rwlock_tryrdlock(&(leaf->key_array_rw_lock))) {
      auto parent = leaf->parent_;
      rcu_progress(worker_id);
#if PROFILE
      auto get_payload_from_parent_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::fgTimeUnit>(get_payload_from_parent_end_time - get_payload_from_parent_start_time).count();
      if (last_parent == superroot_) {
        profileStats.get_payload_from_superroot_fail_time[worker_id] += elapsed_time;
        profileStats.get_payload_superroot_fail_cnt[worker_id]++;
        profileStats.max_get_payload_from_superroot_fail_time[worker_id] =
          std::max(profileStats.max_get_payload_from_superroot_fail_time[worker_id], elapsed_time);
        profileStats.min_get_payload_from_superroot_fail_time[worker_id] =
          std::min(profileStats.min_get_payload_from_superroot_fail_time[worker_id], elapsed_time);
      }
      else {
        profileStats.get_payload_from_parent_fail_time[worker_id] += elapsed_time;
        profileStats.get_payload_directp_fail_cnt[worker_id]++;
        profileStats.max_get_payload_from_parent_fail_time[worker_id] =
          std::max(profileStats.max_get_payload_from_parent_fail_time[worker_id], elapsed_time);
        profileStats.min_get_payload_from_parent_fail_time[worker_id] =
          std::min(profileStats.min_get_payload_from_parent_fail_time[worker_id], elapsed_time);
      }
#endif
      return {1, 0, parent};
    }
    int idx = leaf->find_key(key, worker_id);
    
    if (idx < 0) {
      pthread_rwlock_unlock(&(leaf->key_array_rw_lock));
      auto last_parent = leaf->parent_;
      rcu_progress(worker_id);
      return {1, 0, last_parent};
    } else {
      P rval = leaf->get_payload(idx);
      pthread_rwlock_unlock(&(leaf->key_array_rw_lock));
      rcu_progress(worker_id);
#if PROFILE
      auto get_payload_from_parent_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::fgTimeUnit>(get_payload_from_parent_end_time - get_payload_from_parent_start_time).count();
      if (last_parent == superroot_) {
        profileStats.get_payload_from_superroot_success_time[worker_id] += elapsed_time;
        profileStats.get_payload_superroot_success_cnt[worker_id]++;
        profileStats.max_get_payload_from_superroot_success_time[worker_id] =
          std::max(profileStats.max_get_payload_from_superroot_success_time[worker_id], elapsed_time);
        profileStats.min_get_payload_from_superroot_success_time[worker_id] =
          std::min(profileStats.min_get_payload_from_superroot_success_time[worker_id], elapsed_time);
      }
      else {
        profileStats.get_payload_from_parent_success_time[worker_id] += elapsed_time;
        profileStats.get_payload_directp_success_cnt[worker_id]++;
        profileStats.max_get_payload_from_parent_success_time[worker_id] =
          std::max(profileStats.max_get_payload_from_parent_success_time[worker_id], elapsed_time);
        profileStats.min_get_payload_from_parent_success_time[worker_id] =
          std::min(profileStats.min_get_payload_from_parent_success_time[worker_id], elapsed_time);
      }
#endif
      return {0, rval, nullptr};
    }
  }

  // Looks for the last key no greater than the input value
  // Conceptually, this is equal to the last key before upper_bound()
  // returns end iterator on error
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator find_last_no_greater_than(const AlexKey<T>& key) {
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return end();}
    const int idx = leaf->upper_bound(key) - 1;
    if (idx >= 0) {
      return Iterator(leaf, idx);
    }

    // Edge case: need to check previous data node(s)
    while (true) {
      if (leaf->prev_leaf_.val_ == nullptr) {
        return Iterator(leaf, 0);
      }
      leaf = leaf->prev_leaf_.val_;
      if (leaf->num_keys_ > 0) {
        return Iterator(leaf, leaf->last_pos());
      }
    }
  }

  typename self_type::Iterator begin() {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_[0];
    }
    return Iterator(static_cast<data_node_type*>(cur), 0);
  }

  typename self_type::Iterator end() {
    Iterator it = Iterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  typename self_type::ConstIterator cbegin() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_[0];
    }
    return ConstIterator(static_cast<data_node_type*>(cur), 0);
  }

  typename self_type::ConstIterator cend() const {
    ConstIterator it = ConstIterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  typename self_type::ReverseIterator rbegin() {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      auto model_node = static_cast<model_node_type*>(cur);
      cur = model_node->children_[model_node->num_children_ - 1];
    }
    auto data_node = static_cast<data_node_type*>(cur);
    return ReverseIterator(data_node, data_node->data_capacity_ - 1);
  }

  typename self_type::ReverseIterator rend() {
    ReverseIterator it = ReverseIterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  typename self_type::ConstReverseIterator crbegin() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      auto model_node = static_cast<model_node_type*>(cur);
      cur = model_node->children_[model_node->num_children_ - 1];
    }
    auto data_node = static_cast<data_node_type*>(cur);
    return ConstReverseIterator(data_node, data_node->data_capacity_ - 1);
  }

  typename self_type::ConstReverseIterator crend() const {
    ConstReverseIterator it = ConstReverseIterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  /*** Insert ***/

 public:
  std::pair<Iterator, bool> insert(const V& value, uint32_t worker_id) {
    return insert(value.first, value.second, worker_id);
  }

  template <class InputIterator>
  void insert(InputIterator first, InputIterator last, uint32_t worker_id) {
    for (auto it = first; it != last; ++it) {
      insert(*it, worker_id);
    }
  }

  std::tuple<Iterator, bool, model_node_type *> insert(const AlexKey<T>& key, const P& payload, uint32_t worker_id) {
    return insert_from_parent(key, payload, superroot_, worker_id);
  }

  // This will NOT do an update of an existing key.
  // To perform an update or read-modify-write, do a lookup and modify the
  // payload's value.
  // Returns iterator to inserted element, and whether the insert happened or
  // not.
  // Insert does not happen if duplicates are not allowed and duplicate is
  // found.
  // If it failed finding a leaf, it returns iterator with null leaf with 0 index.
  // If we need to retry later, it returns iterator with null leaf with 1 index
  std::tuple<Iterator, bool, model_node_type *> insert_from_parent(const AlexKey<T>& key, const P& payload, 
                                               model_node_type *last_parent, uint32_t worker_id) {
    // in string ALEX, keys should not fall outside the key domain
#if PROFILE
    if (last_parent == superroot_) {
      profileStats.insert_superroot_call_cnt[worker_id]++;
    }
    else {
      profileStats.insert_directp_call_cnt[worker_id]++;
    }
    auto insert_from_parent_start_time = std::chrono::high_resolution_clock::now();
#endif
    char larger_key = 0;
    char smaller_key = 0;
    for (unsigned int i = 0; i < max_key_length_; i++) {
      if (key.key_arr_[i] > istats_.key_domain_max_[i]) {larger_key = 1; break;}
      else if (key.key_arr_[i] < istats_.key_domain_min_[i]) {smaller_key = 1; break;}
    }
    if (larger_key || smaller_key) {
      std::cout << "worker id : " << worker_id 
                << " root expansion should not happen." << std::endl;
      abort();
    }

    std::vector<TraversalNode<T, P>> traversal_path;
    data_node_type* leaf = get_leaf_from_parent(key, worker_id, last_parent, 1, &traversal_path);
    if (leaf == nullptr) {
      //failed finding leaf, shouldn't happen in normal cases.
      rcu_progress(worker_id);
      return {Iterator(nullptr, 0), false, nullptr};
    } 
    
    model_node_type *parent = traversal_path.back().node;
    if (pthread_mutex_trylock(&leaf->insert_mutex)) {
      //failed obtaining mutex
      rcu_progress(worker_id);
#if PROFILE
      auto insert_from_parent_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::fgTimeUnit>(insert_from_parent_end_time - insert_from_parent_start_time).count();
      if (last_parent == superroot_) {
        profileStats.insert_from_superroot_fail_time[worker_id] += elapsed_time;
        profileStats.insert_superroot_fail_cnt[worker_id]++;
        profileStats.max_insert_from_superroot_fail_time[worker_id] =
          std::max(profileStats.max_insert_from_superroot_fail_time[worker_id], elapsed_time);
        profileStats.min_insert_from_superroot_fail_time[worker_id] =
          std::min(profileStats.min_insert_from_superroot_fail_time[worker_id], elapsed_time);
      }
      else {
        profileStats.insert_from_parent_fail_time[worker_id] += elapsed_time;
        profileStats.insert_directp_fail_cnt[worker_id]++;
        profileStats.max_insert_from_parent_fail_time[worker_id] =
          std::max(profileStats.max_insert_from_parent_fail_time[worker_id], elapsed_time);
        profileStats.min_insert_from_parent_fail_time[worker_id] =
          std::min(profileStats.min_insert_from_parent_fail_time[worker_id], elapsed_time);
      }
#endif
      return {Iterator(nullptr, 1), false, parent};
    }
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - in final, decided to insert at bucketID : "
              << traversal_path.back().bucketID << std::endl;
    alex::coutLock.unlock();
#endif

    // Nonzero fail flag means that the insert did not happen
    std::pair<std::pair<int, int>, std::pair<data_node_type *, data_node_type *>> ret 
      = leaf->insert(key, payload, worker_id);
    int fail = ret.first.first;
    int insert_pos = ret.first.second;
    leaf = ret.second.first;
    //data_node_type *maybe_new_data_node = ret.second.second;

    if (fail == -1) {
      // Duplicate found and duplicates not allowed
      pthread_mutex_unlock(&leaf->insert_mutex);
      memory_fence();
      rcu_progress(worker_id);
      return {Iterator(leaf, insert_pos), false, nullptr};
    }
    else if (!fail) {//succeded without modification
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "alex.h insert : succeeded insertion and processing" << std::endl;
      alex::coutLock.unlock();
#endif
      pthread_mutex_unlock(&leaf->insert_mutex);
      memory_fence();
      num_keys.increment();
      rcu_progress(worker_id);
#if PROFILE
      auto insert_from_parent_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::fgTimeUnit>(insert_from_parent_end_time - insert_from_parent_start_time).count();
      if (last_parent == superroot_) {
        profileStats.insert_from_superroot_success_time[worker_id] += elapsed_time;
        profileStats.insert_superroot_success_cnt[worker_id]++;
        profileStats.max_insert_from_superroot_success_time[worker_id] =
          std::max(profileStats.max_insert_from_superroot_success_time[worker_id], elapsed_time);
        profileStats.min_insert_from_superroot_success_time[worker_id] =
          std::min(profileStats.min_insert_from_superroot_success_time[worker_id], elapsed_time);
      }
      else {
        profileStats.insert_from_parent_success_time[worker_id] += elapsed_time;
        profileStats.insert_directp_success_cnt[worker_id]++;
        profileStats.max_insert_from_parent_success_time[worker_id] =
          std::max(profileStats.max_insert_from_parent_success_time[worker_id], elapsed_time);
        profileStats.min_insert_from_parent_success_time[worker_id] =
          std::min(profileStats.min_insert_from_parent_success_time[worker_id], elapsed_time);
      }
#endif
      return {Iterator(leaf, insert_pos), true, nullptr}; //iterator could be invalid.
    }
    else { //succeeded, but needs to modify
      if (fail == 4) { //need to expand
        if (cur_bg_num.load() < config.max_bgnum) { //but only when we're not runnign max bg threads
          cur_bg_num++;
          memory_fence();
          expandParam *param = new expandParam();
          param->leaf = leaf;
          param->worker_id = worker_id;
          pthread_t pthread;

          pthread_create(&pthread, nullptr, expand_handler, (void *)param);
          pthread_detach(pthread); //detach since it's not joined.
        }
        else {pthread_mutex_unlock(&leaf->insert_mutex);}
      }
      else {
        if (fail == 5 || cur_bg_num.load() < config.max_bgnum) {
          //create thread that handles modification and let it handle
          cur_bg_num++;
          memory_fence();
          alexIParam *param = new alexIParam();
          param->leaf = leaf;
          param->worker_id = worker_id;
          param->bucketID = traversal_path.back().bucketID;
          param->fail = fail;
          param->this_ptr = this;
          pthread_t pthread;

          pthread_create(&pthread, nullptr, insert_fail_handler, (void *)param);   
          pthread_detach(pthread); //detach since it's not joined
        }
        else {pthread_mutex_unlock(&leaf->insert_mutex);}
      }

      //original thread returns and retry later. (need to rcu_progress)
      rcu_progress(worker_id);

#if PROFILE
      auto insert_from_parent_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::fgTimeUnit>(insert_from_parent_end_time - insert_from_parent_start_time).count();

      if (last_parent == superroot_) {
        profileStats.insert_from_superroot_success_time[worker_id] += elapsed_time;
        profileStats.insert_superroot_success_cnt[worker_id]++;
        profileStats.max_insert_from_superroot_success_time[worker_id] =
          std::max(profileStats.max_insert_from_superroot_success_time[worker_id], elapsed_time);
        profileStats.min_insert_from_superroot_success_time[worker_id] =
          std::min(profileStats.min_insert_from_superroot_success_time[worker_id], elapsed_time);
      }
      else {
        profileStats.insert_from_parent_success_time[worker_id] += elapsed_time;
        profileStats.insert_directp_success_cnt[worker_id]++;
        profileStats.max_insert_from_parent_success_time[worker_id] =
          std::max(profileStats.max_insert_from_parent_success_time[worker_id], elapsed_time);
        profileStats.min_insert_from_parent_success_time[worker_id] =
          std::min(profileStats.min_insert_from_parent_success_time[worker_id], elapsed_time);
      }
#endif

      return {Iterator(leaf, insert_pos), true, nullptr}; //iterator could be invalid.
    }
  }

 private:
  struct expandParam {
    data_node_type *leaf;
    uint32_t worker_id;;
  };

  struct alexIParam {
    data_node_type *leaf;
    uint32_t worker_id;
    int bucketID;
    int fail;
    self_type *this_ptr;
  };

  static void *expand_handler(void *param) {
    expandParam *Eparam = (expandParam *)param;
    data_node_type *leaf = Eparam->leaf;
    uint32_t worker_id = Eparam->worker_id;

#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - failed and made a thread to modify node" << std::endl;
    std::cout << "parent is : " << leaf->parent_ << std::endl;
    alex::coutLock.unlock();
#endif

    leaf->resize(data_node_type::kMinDensity_, false,
                  leaf->is_append_mostly_right(),
                  leaf->is_append_mostly_left());

#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << leaf->parent_ << " - ";
    std::cout << "alex.h expanded data node" << std::endl;
    alex::coutLock.unlock();
#endif

    //will use the original data node!
    pthread_mutex_unlock(&leaf->insert_mutex);
    memory_fence();
    delete Eparam;
    cur_bg_num--;
    memory_fence();
    pthread_exit(nullptr);
  }

  static void *insert_fail_handler(void *param) {
    //parameter obtaining
    alexIParam *Iparam = (alexIParam *) param;
    data_node_type *leaf = Iparam->leaf;
    uint32_t worker_id = Iparam->worker_id;
    int bucketID = Iparam->bucketID;
    int fail = Iparam->fail;
    self_type *this_ptr = Iparam->this_ptr;

    model_node_type* parent = leaf->parent_;
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - failed and made a thread to modify node\n";
    std::cout << "parent is : " << parent << '\n'
    std::cout << "bucketID : " << bucketID << std::endl;
    alex::coutLock.unlock();
#endif

    std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;

    int fanout_tree_depth = 1;
    double *model_param = nullptr;
    auto ret = fanout_tree::find_best_fanout_existing_node<T, P>(
          leaf, this_ptr->num_keys.read(), used_fanout_tree_nodes, 2, worker_id);
    fanout_tree_depth = ret.first;
    model_param = ret.second;
              
    int best_fanout = 1 << fanout_tree_depth;

    if (fanout_tree_depth == 0) {
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
      std::cout << "failed and decided to expand" << std::endl;
      alex::coutLock.unlock();
#endif
      // expand existing data node and retrain model
      leaf->resize(data_node_type::kMinDensity_, true,
                   leaf->is_append_mostly_right(),
                   leaf->is_append_mostly_left());
      fanout_tree::FTNode& tree_node = used_fanout_tree_nodes[0];
      leaf->cost_ = tree_node.cost;
      leaf->expected_avg_exp_search_iterations_ =
          tree_node.expected_avg_search_iterations;
      leaf->expected_avg_shifts_ = tree_node.expected_avg_shifts;
      leaf->reset_stats();

      pthread_mutex_unlock(&leaf->insert_mutex);
      memory_fence();
    } else {
      bool reuse_model = (fail == 3);
      // either split sideways or downwards
      // synchronization is covered automatically in splitting functions.
      bool should_split_downwards =
          (parent->num_children_ * best_fanout /
                   (1 << leaf->duplication_factor_) >
               this_ptr->derived_params_.max_fanout ||
           parent->level_ == this_ptr->superroot_->level_ ||
           (fanout_tree_depth > leaf->duplication_factor_));
      if (should_split_downwards) {
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
        std::cout << "failed and decided to split downwards" << std::endl;
        alex::coutLock.unlock();
#endif
        split_downwards(parent, bucketID, fanout_tree_depth, model_param, used_fanout_tree_nodes,
                                 reuse_model, worker_id, this_ptr);
      } else {
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
        std::cout << "failed and decided to split sideways" << std::endl;
        alex::coutLock.unlock();
#endif
        split_sideways(parent, bucketID, fanout_tree_depth, used_fanout_tree_nodes,
                       reuse_model, worker_id, this_ptr);
      }
    }

    delete[] ret.second;

    //empty used_fanout_tree_nodes for preventing memory leakage.
    for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {delete[] tree_node.a;}

    //return successfully.
    delete Iparam;
    cur_bg_num--;
    memory_fence();
    pthread_exit(nullptr);
  }

  // Splits downwards in the manner determined by the fanout tree and updates
  // the pointers of the parent.
  // If no fanout tree is provided, then splits downward in two. Returns the
  // newly created model node.
  static void split_downwards(
      model_node_type* parent, int bucketID, int fanout_tree_depth, double *model_param,
      std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
      bool reuse_model, uint32_t worker_id, self_type *this_ptr) {
#if PROFILE
    profileStats.split_downwards_call_cnt++;
    auto split_downwards_start_time = std::chrono::high_resolution_clock::now();
#endif
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "...bucketID : " << bucketID << std::endl;
    alex::coutLock.unlock();
#endif
    auto leaf = static_cast<data_node_type*> (parent->children_[bucketID]);
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "and leaf is : " << leaf << std::endl;
    alex::coutLock.unlock();
#endif

    // Create the new model node that will replace the current data node
    int fanout = 1 << fanout_tree_depth;
    auto new_node = new (this_ptr->model_node_allocator().allocate(1))
        model_node_type(leaf->level_, parent, this_ptr->allocator_);
    new_node->duplication_factor_ = leaf->duplication_factor_;
    new_node->num_children_ = fanout;
    new_node->children_ = new node_type*[fanout];
    //needs to initialize min/max key in case of split_downwards.
    std::copy(leaf->min_key_.val_->key_arr_, leaf->min_key_.val_->key_arr_ + max_key_length_,
              new_node->min_key_.val_->key_arr_);
    std::copy(leaf->max_key_.val_->key_arr_, leaf->max_key_.val_->key_arr_ + max_key_length_,
              new_node->max_key_.val_->key_arr_);


    int repeats = 1 << leaf->duplication_factor_;
    int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child
    int end_bucketID =
        start_bucketID + repeats;  // first bucket with different child

    std::copy(model_param, model_param + max_key_length_, new_node->model_.a_);
    new_node->model_.b_ = model_param[max_key_length_];

#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << "'s generated thread - ";
    //std::cout << "left prediction result (sd) " << new_node->model_.predict_double(leaf->key_slots_[leaf->first_pos()]) << std::endl;
    //std::cout << "right prediction result (sd) " << new_node->model_.predict_double(leaf->key_slots_[leaf->last_pos()]) << std::endl;
    //alex::coutLock.unlock();
#endif

    // Create new data nodes
    if (used_fanout_tree_nodes.empty()) {
      assert(fanout_tree_depth == 1);
      create_two_new_data_nodes(leaf, new_node, fanout_tree_depth,
                                                    reuse_model, worker_id, this_ptr);
    } else {
      create_new_data_nodes(leaf, new_node, fanout_tree_depth,
                            used_fanout_tree_nodes, worker_id, this_ptr);
    }

    //substitute pointers in parent model node
    pthread_rwlock_wrlock(&(parent->children_rw_lock_));
    for (int i = start_bucketID; i < end_bucketID; i++) {
      parent->children_[i] = new_node;
    }
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "split_downwards parent children_\n";
    for (int i = 0; i < parent->num_children_; i++) {
      std::cout << i << " : " << parent_new_children[i] << '\n';
    }
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "min_key_(model_node) : " << new_node->min_key_.val_->key_arr_ << '\n';
    std::cout << "max_key_(model_node) : " << new_node->max_key_.val_->key_arr_ << '\n';
    for (int i = 0; i < fanout; i++) {
        std::cout << i << "'s min_key is : "
                  << new_node->children_[i]->min_key_.val_->key_arr_ << '\n';
        std::cout << i << "'s max_key is : " 
                  << new_node->children_[i]->max_key_.val_->key_arr_ << '\n';
    }
    std::cout << std::flush;
    alex::coutLock.unlock();
#endif
    pthread_rwlock_unlock(&(parent->children_rw_lock_));
    if (parent == this_ptr->superroot_) {
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - "
                << "root node splitted downwards" << std::endl;
      alex::coutLock.unlock();
#endif
      this_ptr->root_node_ = new_node;
    }

    //destroy unused leaf and metadata after waiting.
    rcu_barrier();
    this_ptr->delete_node(leaf);
#if PROFILE
    auto split_downwards_end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::bgTimeUnit>(split_downwards_end_time - split_downwards_start_time).count();
    profileStats.split_downwards_time += elapsed_time;
    profileStats.max_split_downwards_time =
      std::max(profileStats.max_split_downwards_time.load(), elapsed_time);
    profileStats.min_split_downwards_time =
      std::min(profileStats.min_split_downwards_time.load(), elapsed_time);
#endif
  }

  // Splits data node sideways in the manner determined by the fanout tree.
  // If no fanout tree is provided, then splits sideways in two.
  static void split_sideways(model_node_type* parent, int bucketID,
                      int fanout_tree_depth,
                      std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
                      bool reuse_model, uint32_t worker_id, self_type *this_ptr) {
#if PROFILE
    profileStats.split_sideways_call_cnt++;
    auto split_sideways_start_time = std::chrono::high_resolution_clock::now();
#endif
    auto leaf = static_cast<data_node_type*>(parent->children_[bucketID]);

    int fanout = 1 << fanout_tree_depth;
    int repeats = 1 << leaf->duplication_factor_;
    if (fanout > repeats) {
      //in multithreading, because of synchronization issue of duplication_fcator_
      //we don't do model expansion.
      ;
    }
    int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child

    if (used_fanout_tree_nodes.empty()) {
      assert(fanout_tree_depth == 1);
      create_two_new_data_nodes(leaf, parent,
          std::max(fanout_tree_depth, static_cast<int>(leaf->duplication_factor_)),
          reuse_model, worker_id, this_ptr, start_bucketID);
    } else {
      // Extra duplication factor is required when there are more redundant
      // pointers than necessary
      int extra_duplication_factor =
          std::max(0, leaf->duplication_factor_ - fanout_tree_depth);
      create_new_data_nodes(leaf, parent, fanout_tree_depth,
                            used_fanout_tree_nodes, worker_id, this_ptr,
                            start_bucketID, extra_duplication_factor);
    }

    rcu_barrier();
    this_ptr->delete_node(leaf);
#if PROFILE
    auto split_sideways_end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::bgTimeUnit>(split_sideways_end_time - split_sideways_start_time).count();
    profileStats.split_sideways_time += elapsed_time;
    profileStats.max_split_sideways_time =
      std::max(profileStats.max_split_sideways_time.load(), elapsed_time);
    profileStats.min_split_sideways_time =
      std::min(profileStats.min_split_sideways_time.load(), elapsed_time);
#endif
  }

  // Create two new data nodes by equally dividing the key space of the old data
  // node, insert the new
  // nodes as children of the parent model node starting from a given position,
  // and link the new data nodes together.
  // duplication_factor denotes how many child pointer slots were assigned to
  // the old data node.
  // returns destroy needed old meta data.
  static void create_two_new_data_nodes(data_node_type* old_node,
                                 model_node_type* parent, int duplication_factor, 
                                 bool reuse_model, uint32_t worker_id,
                                 self_type *this_ptr, int start_bucketID = 0) {
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    //std::cout << "called create_two_new_dn" << std::endl;
    //alex::coutLock.unlock();
#endif
    assert(duplication_factor >= 1);
    int num_buckets = 1 << duplication_factor;
    int end_bucketID = start_bucketID + num_buckets;
    int mid_bucketID = start_bucketID + num_buckets / 2;

    bool append_mostly_right = old_node->is_append_mostly_right();
    int appending_right_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->max_key_.val_)), 0),
        parent->num_children_ - 1);
    bool append_mostly_left = old_node->is_append_mostly_left();
    int appending_left_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->min_key_.val_)), 0),
        parent->num_children_ - 1);

    int right_boundary = 0;
    AlexKey<T> tmpkey;
    tmpkey.key_arr_ = new T[max_key_length_];
    //According to my insight, linear model would be monotonically increasing
    //So I think we could compute key corresponding to mid_bucketID as
    //average of min/max key of current splitting node.
    for (unsigned int i = 0; i < max_key_length_; i++) {
      tmpkey.key_arr_[i] = (old_node->max_key_.val_->key_arr_[i] + old_node->min_key_.val_->key_arr_[i]) / 2;
    }
    
    right_boundary = old_node->lower_bound(tmpkey);
    // Account for off-by-one errors due to floating-point precision issues.
    while (right_boundary < old_node->data_capacity_) {
      AlexKey<T> old_rbkey = old_node->get_key(right_boundary);
      if (this_ptr->key_equal(old_rbkey, old_node->kEndSentinel_)) {break;}
      if (parent->model_.predict(old_node->get_key(right_boundary)) >= mid_bucketID) {break;}
      right_boundary = std::min(
          old_node->get_next_filled_position(right_boundary, false) + 1,
          old_node->data_capacity_);
    }
    data_node_type* left_leaf = bulk_load_leaf_node_from_existing(
        old_node, 0, right_boundary, worker_id, this_ptr, true, nullptr, reuse_model,
        append_mostly_right && start_bucketID <= appending_right_bucketID &&
            appending_right_bucketID < mid_bucketID,
        append_mostly_left && start_bucketID <= appending_left_bucketID &&
            appending_left_bucketID < mid_bucketID);
    data_node_type* right_leaf = bulk_load_leaf_node_from_existing(
        old_node, right_boundary, old_node->data_capacity_, worker_id, this_ptr, true, nullptr, reuse_model,
        append_mostly_right && mid_bucketID <= appending_right_bucketID &&
            appending_right_bucketID < end_bucketID,
        append_mostly_left && mid_bucketID <= appending_left_bucketID &&
            appending_left_bucketID < end_bucketID);
    old_node->pending_left_leaf_.update(left_leaf);
    old_node->pending_right_leaf_.update(right_leaf);
    left_leaf->level_ = static_cast<short>(parent->level_ + 1);
    right_leaf->level_ = static_cast<short>(parent->level_ + 1);
    left_leaf->duplication_factor_ =
        static_cast<uint8_t>(duplication_factor - 1);
    right_leaf->duplication_factor_ =
        static_cast<uint8_t>(duplication_factor - 1);
    left_leaf->parent_ = parent;
    right_leaf->parent_ = parent;

    pthread_rwlock_wrlock(&(parent->children_rw_lock_));
    for (int i = start_bucketID; i < mid_bucketID; i++) {
      parent->children_[i] = left_leaf;
    }
    for (int i = mid_bucketID; i < end_bucketID; i++) {
      parent->children_[i] = right_leaf;
    }
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
      std::cout << "two new data node made with left min/max as "
                << left_leaf->min_key_.val_->key_arr_
                << " " << left_leaf->max_key_.val_->key_arr_
                << "and right min/max as "
                << right_leaf->min_key_.val_->key_arr_
                << " " << right_leaf->max_key_.val_->key_arr_
                << std::endl;
      alex::coutLock.unlock();
#endif
    pthread_rwlock_unlock(&(parent->children_rw_lock_));
    this_ptr->link_data_nodes(old_node, left_leaf, right_leaf);
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    //std::cout << "finished create_two_new_dn" << std::endl;
    //alex::coutLock.unlock();
#endif
  }

  // Create new data nodes from the keys in the old data node according to the
  // fanout tree, insert the new
  // nodes as children of the parent model node starting from a given position,
  // and link the new data nodes together.
  // Helper for splitting when using a fanout tree.
  // returns destroy needed old meta data.
 static void create_new_data_nodes(
      data_node_type* old_node, model_node_type* parent,
      int fanout_tree_depth, std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
      uint32_t worker_id, self_type *this_ptr, 
      int start_bucketID = 0, int extra_duplication_factor = 0) {
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    //std::cout << "called create_new_dn" << std::endl;
    //std::cout << "old node is " << old_node << std::endl;
    //alex::coutLock.unlock();
#endif
    bool append_mostly_right = old_node->is_append_mostly_right();
    int appending_right_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->max_key_.val_)), 0),
        parent->num_children_ - 1);
    bool append_mostly_left = old_node->is_append_mostly_left();
    int appending_left_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->min_key_.val_)), 0),
        parent->num_children_ - 1);

    // Create the new data nodes
    int cur = start_bucketID;  // first bucket with same child
    std::vector<std::pair<node_type *, int>> generated_nodes;
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "starting bucket is" << start_bucketID << std::endl;
    alex::coutLock.unlock();
#endif
    data_node_type* prev_leaf =
        old_node->prev_leaf_.read();  // used for linking the new data nodes
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "initial prev_leaf is : " << prev_leaf << std::endl;
    alex::coutLock.unlock();
#endif
    int left_boundary = 0;
    int right_boundary = 0;
    // Keys may be re-assigned to an adjacent fanout tree node due to off-by-one
    // errors
    int num_reassigned_keys = 0;
    int first_iter = 1;
    for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
      left_boundary = right_boundary;
      auto duplication_factor = static_cast<uint8_t>(
          fanout_tree_depth - tree_node.level + extra_duplication_factor);
      int child_node_repeats = 1 << duplication_factor;
      bool keep_left = append_mostly_right && cur <= appending_right_bucketID &&
                       appending_right_bucketID < cur + child_node_repeats;
      bool keep_right = append_mostly_left && cur <= appending_left_bucketID &&
                        appending_left_bucketID < cur + child_node_repeats;
      right_boundary = tree_node.right_boundary;
      // Account for off-by-one errors due to floating-point precision issues.
      tree_node.num_keys -= num_reassigned_keys;
      num_reassigned_keys = 0;
      while (right_boundary < old_node->data_capacity_) {
        AlexKey<T> old_node_rbkey = old_node->get_key(right_boundary);
        if (this_ptr->key_equal(old_node_rbkey, old_node->kEndSentinel_)) {break;}
        if (parent->model_.predict(old_node->get_key(right_boundary)) >=
                 cur + child_node_repeats) {break;}
        num_reassigned_keys++;
        right_boundary = std::min(
            old_node->get_next_filled_position(right_boundary, false) + 1,
            old_node->data_capacity_);
      }
      tree_node.num_keys += num_reassigned_keys;
      data_node_type* child_node = bulk_load_leaf_node_from_existing(
          old_node, left_boundary, right_boundary, worker_id, this_ptr, false, &tree_node, false,
          keep_left, keep_right);
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
      std::cout << "child_node pointer : " << child_node << std::endl;
      alex::coutLock.unlock();
#endif
      child_node->level_ = static_cast<short>(parent->level_ + 1);
      child_node->cost_ = tree_node.cost;
      child_node->duplication_factor_ = duplication_factor;
      child_node->expected_avg_exp_search_iterations_ =
          tree_node.expected_avg_search_iterations;
      child_node->expected_avg_shifts_ = tree_node.expected_avg_shifts;

      if (first_iter) { //left leaf is not a new data node
        old_node->pending_left_leaf_.update(child_node);
#if DEBUG_PRINT
        //alex::coutLock.lock();
        //std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
        //std::cout << "updated pll with " << child_node << std::endl;
        //alex::coutLock.unlock();
#endif
        if (prev_leaf != nullptr) {
          data_node_type *prev_leaf_pending_rl = prev_leaf->pending_right_leaf_.read();
          if (prev_leaf_pending_rl != nullptr) {
            child_node->prev_leaf_.update(prev_leaf_pending_rl);
            prev_leaf_pending_rl->next_leaf_.update(child_node);
          }
          else {
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
            std::cout << "child_node's prev_leaf_ is " << prev_leaf << std::endl;
            alex::coutLock.unlock();
#endif
            child_node->prev_leaf_.update(prev_leaf);
            prev_leaf->next_leaf_.update(child_node);
          }
        }
        else {
          child_node->prev_leaf_.update(nullptr);
        }
        first_iter = 0;
      }
      else {
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
        std::cout << "child_node's prev_leaf_ is " << prev_leaf << std::endl;
        alex::coutLock.unlock();
#endif
        child_node->prev_leaf_.update(prev_leaf);
        prev_leaf->next_leaf_.update(child_node);
      }
      child_node->parent_ = parent;
      cur += child_node_repeats;
      generated_nodes.push_back({child_node, child_node_repeats});
      prev_leaf = child_node;
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
      std::cout << "new data node made with min_key as "
                << child_node->min_key_.val_->key_arr_
                << " and max_key as "
                << child_node->max_key_.val_->key_arr_
                << std::endl;
      alex::coutLock.unlock();
#endif
    }
    pthread_rwlock_wrlock(&(parent->children_rw_lock_));
    cur = start_bucketID;
    //update model node metadata
    for (auto it = generated_nodes.begin(); it != generated_nodes.end(); ++it) {
      auto generated_node = *it;
      for (int i = cur; i < cur + generated_node.second; ++i) {
        parent->children_[i] = generated_node.first;
      }
      cur += generated_node.second;
    }
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
      std::cout << "cndn children_\n";
      for (int i = 0 ; i < parent->num_children_; i++) {
        std::cout << i << " : " << parent->children_[i] << '\n';
      }
      std::cout << std::flush;
      alex::coutLock.unlock();
#endif
    pthread_rwlock_unlock(&(parent->children_rw_lock_));

    //update right-most leaf's next/prev leaf.
    old_node->pending_right_leaf_.update(prev_leaf);
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    //std::cout << "updated prl with " << prev_leaf << std::endl;
    //alex::coutLock.unlock();
#endif
    data_node_type *next_leaf = old_node->next_leaf_.read();
    if (next_leaf != nullptr) {
      data_node_type *next_leaf_pending_ll = next_leaf->pending_left_leaf_.read();
      if (next_leaf_pending_ll != nullptr) {
        prev_leaf->next_leaf_.update(next_leaf_pending_ll);
        next_leaf_pending_ll->prev_leaf_.update(prev_leaf);
      }
      else {
        prev_leaf->next_leaf_.update(next_leaf);
        next_leaf->prev_leaf_.update(prev_leaf);
      }
    }
    else {
      prev_leaf->next_leaf_.update(nullptr);
    }
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << "'s generated thread for parent " << parent << " - ";
    std::cout << "finished create_new_dn" << std::endl;
    alex::coutLock.unlock();
#endif
  }

  /*** Stats ***/

 public:
  // Number of elements
  size_t size() { return static_cast<size_t>(num_keys.read()); }

  // True if there are no elements
  bool empty() const { return (size() == 0); }

  // This is just a function required by the STL standard. ALEX can hold more
  // items.
  size_t max_size() const { return size_t(-1); }

  // Size in bytes of all the keys, payloads, and bitmaps stored in this index
  long long data_size() const {
    long long size = 0;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      node_type* cur = node_it.current();
      if (cur->is_leaf_) {
        size += static_cast<data_node_type*>(cur)->data_size();
      }
    }
    return size;
  }

  // Size in bytes of all the model nodes (including pointers) and metadata in
  // data nodes
  // should only be called when alex structure is not being modified.
  long long model_size() const {
    long long size = 0;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      size += node_it.current()->node_size();
    }
    return size;
  }

  /*** Iterators ***/

 public:
  class Iterator {
   public:
    data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    Iterator() {}

    Iterator(data_node_type* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    Iterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    Iterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    Iterator& operator=(const Iterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    Iterator& operator++() {
      advance();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_].val_);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const {return ((data_node_type *) cur_leaf_)->get_key(cur_idx_); }

    P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const Iterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_.read();
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
          cur_leaf_ = cur_leaf_->next_leaf_.read();
          cur_idx_ = 0;
          if (cur_leaf_ == nullptr) {
            return;
          }
          cur_bitmap_idx_ = 0;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }
  };

  class ConstIterator {
   public:
    const data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    ConstIterator() {}

    ConstIterator(const data_node_type* leaf, int idx)
        : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    ConstIterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstIterator(const ConstIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstIterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstIterator(const ConstReverseIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstIterator& operator=(const ConstIterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    ConstIterator& operator++() {
      advance();
      return *this;
    }

    ConstIterator operator++(int) {
      ConstIterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    const V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const { return cur_leaf_->get_key(cur_idx_); }

    const P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const ConstIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const ConstIterator& rhs) const { return !(*this == rhs); };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
          cur_leaf_ = cur_leaf_->next_leaf_;
          cur_idx_ = 0;
          if (cur_leaf_ == nullptr) {
            return;
          }
          cur_bitmap_idx_ = 0;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }
  };

  class ReverseIterator {
   public:
    data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    ReverseIterator() {}

    ReverseIterator(data_node_type* leaf, int idx)
        : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    ReverseIterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ReverseIterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ReverseIterator& operator=(const ReverseIterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    ReverseIterator& operator++() {
      advance();
      return *this;
    }

    ReverseIterator operator++(int) {
      ReverseIterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const { return cur_leaf_->get_key(cur_idx_); }

    P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const ReverseIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const ReverseIterator& rhs) const {
      return !(*this == rhs);
    };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= (1ULL << bit_pos) | ((1ULL << bit_pos) - 1);

      advance();
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_--;
        if (cur_bitmap_idx_ < 0) {
          cur_leaf_ = cur_leaf_->prev_leaf_.read();
          if (cur_leaf_ == nullptr) {
            cur_idx_ = 0;
            return;
          }
          cur_idx_ = cur_leaf_->data_capacity_ - 1;
          cur_bitmap_idx_ = cur_leaf_->bitmap_size_ - 1;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      assert(cpu_supports_bmi());
      int bit_pos = static_cast<int>(63 - _lzcnt_u64(cur_bitmap_data_));
      cur_idx_ = (cur_bitmap_idx_ << 6) + bit_pos;
      cur_bitmap_data_ &= ~(1ULL << bit_pos);
    }
  };

  class ConstReverseIterator {
   public:
    const data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    ConstReverseIterator() {}

    ConstReverseIterator(const data_node_type* leaf, int idx)
        : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    ConstReverseIterator(const ConstReverseIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstReverseIterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstReverseIterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstReverseIterator(const ConstIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstReverseIterator& operator=(const ConstReverseIterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    ConstReverseIterator& operator++() {
      advance();
      return *this;
    }

    ConstReverseIterator operator++(int) {
      ConstReverseIterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    const V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const { return cur_leaf_->get_key(cur_idx_); }

    const P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const ConstReverseIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const ConstReverseIterator& rhs) const {
      return !(*this == rhs);
    };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= (1ULL << bit_pos) | ((1ULL << bit_pos) - 1);

      advance();
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_--;
        if (cur_bitmap_idx_ < 0) {
          cur_leaf_ = cur_leaf_->prev_leaf_.read();
          if (cur_leaf_ == nullptr) {
            cur_idx_ = 0;
            return;
          }
          cur_idx_ = cur_leaf_->data_capacity_ - 1;
          cur_bitmap_idx_ = cur_leaf_->bitmap_size_ - 1;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      assert(cpu_supports_bmi());
      int bit_pos = static_cast<int>(63 - _lzcnt_u64(cur_bitmap_data_));
      cur_idx_ = (cur_bitmap_idx_ << 6) + bit_pos;
      cur_bitmap_data_ &= ~(1ULL << bit_pos);
    }
  };

  // Iterates through all nodes with pre-order traversal
  class NodeIterator {
   public:
    const self_type* index_;
    node_type* cur_node_;
    std::stack<node_type*> node_stack_;  // helps with traversal

    // Start with root as cur and all children of root in stack
    explicit NodeIterator(const self_type* index)
        : index_(index), cur_node_(index->root_node_) {
      if (cur_node_ && !cur_node_->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur_node_);
        node_stack_.push(node->children_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_[i] != node->children_[i + 1]) {
            node_stack_.push(node->children_[i]);
          }
        }
      }
    }

    node_type* current() const { return cur_node_; }

    node_type* next() {
      if (node_stack_.empty()) {
        cur_node_ = nullptr;
        return nullptr;
      }

      cur_node_ = node_stack_.top();
      node_stack_.pop();

      if (!cur_node_->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur_node_);
        node_stack_.push(node->children_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_[i] != node->children_[i + 1]) {
            node_stack_.push(node->children_[i]);
          }
        }
      }

      return cur_node_;
    }

    bool is_end() const { return cur_node_ == nullptr; }
  };
};
}  // namespace alex
