#pragma once

#include "alex_base.h"

namespace alex {

const uint8_t alt_buf_fanout = 16; //total fanout
const uint8_t node_capacity = alt_buf_fanout - 1; //number of keys that can be handled.

//forward declaration
template <class T, class P> class DeltaNode;
template <class T, class P> class DeltaInternal;
template <class T, class P> class DeltaLeaf;
template <class T, class P> class DeltaIndex;

template <class T, class P>
class DeltaNode {
public:
    typedef DeltaInternal<T, P> internal_t;

    bool is_leaf_ = false;
    internal_t *parent_ = nullptr; //used for recursive splitting
    uint8_t key_n_ = 0; //for boundary in binary search
    uint8_t pos_in_childrens_ = -1; //pos this node locates in childrens_ of parent (if exists)
    AlexKey<T> keys_[node_capacity];
    
public:
    DeltaNode() = default;
    DeltaNode(bool is_leaf) : is_leaf_(is_leaf) {}
    DeltaNode(bool is_leaf, internal_t *parent) : is_leaf_(is_leaf), parent_(parent) {}
    virtual ~DeltaNode() = default;

public:
    //binary searching pos
    //where it contains key that is larger than or equal to the parameter key.
    //if parameter key is larger than any key, returns key_n_.
    uint8_t find_first_larger_than_or_equal_to(const AlexKey<T> &key) {
        uint8_t left_idx = 0;
        uint8_t right_idx = key_n_;
        uint8_t mid = left_idx + (right_idx - left_idx) / 2;
        while (left_idx < right_idx) {
            if (keys_[mid] < key) {
                left_idx = mid + 1;
            } else {
                right_idx = mid;
            }
            mid = left_idx + (right_idx - left_idx) / 2;
        }
        return left_idx;
    }

    //binary searching pos
    //where it contains key that is larger than the parameter key.
    //if parameter key is larger than any key, returns key_n_.
    uint8_t find_first_larger_than(const AlexKey<T> &key) {
        uint8_t left_idx = 0;
        uint8_t right_idx = key_n_;
        uint8_t mid = left_idx + (right_idx - left_idx) / 2;
        while (left_idx < right_idx) {
            if (keys_[mid] <= key) {
                left_idx = mid + 1;
            } else {
                right_idx = mid;
            }
            mid = left_idx + (right_idx - left_idx) / 2;
        }
        return left_idx;
    }

    AlexKey<T>& get_key(uint8_t pos) {
        return keys_[pos];
    }
};

template <class T, class P>
class DeltaInternal : public DeltaNode<T, P> {
public:
    typedef DeltaNode<T, P> node_t;
    typedef DeltaInternal<T, P> internal_t;

    uint8_t child_n_ = 0;
    node_t *childrens_[alt_buf_fanout];

public:
    DeltaInternal() : DeltaNode<T, P>() {}
    DeltaInternal(internal_t *parent) : DeltaNode<T,P>(false, parent) {} 
    ~DeltaInternal() = default;

public:
    node_t *get_children(uint8_t pos) {
        return childrens_[pos];
    }

    node_t *find_child(const AlexKey<T> &key) {
        return childrens_[this->find_first_larger_than(key)];
    }

    //[begin, end] -> [begin+1, end+1]
    void shift_by_one(int begin, int end) {
        assert(end < this->key_n_ && end < (node_capacity - 1));
        for (int i = end; i >= begin; --i) {
            this->keys_[i+1] = this->keys_[i];
            childrens_[i+1] = childrens_[i];
            childrens_[i+1]->pos_in_childrens_ = i+1;
        }
    }

    //[begin, key_n_-1] -> [begin+1, key_n_]
    void shift_by_one(int begin) {
        assert(begin < this->key_n_);
        for (int i = this->key_n_ - 1; i >= begin; --i) {
            this->keys_[i+1] = this->keys_[i];
            childrens_[i+1] = childrens_[i];
            childrens_[i+1]->pos_in_childrens_ = i+1;
        }
    }
};

template <class T, class P>
class DeltaLeaf : public DeltaNode<T, P> {
public:
    typedef DeltaInternal<T, P> internal_t;
    typedef DeltaLeaf<T, P> leaf_t;

    T payloads_[node_capacity];
    leaf_t *next_leaf_ = nullptr;
    leaf_t *prev_leaf_ = nullptr;

public:
    DeltaLeaf() : DeltaNode<T, P>(true) {}
    DeltaLeaf(internal_t *parent) : DeltaNode<T, P>(true, parent) {}
    DeltaLeaf(internal_t *parent, leaf_t *next_leaf) : 
        DeltaNode<T, P>(true, parent), next_leaf_(next_leaf) {}
    ~DeltaLeaf() = default;

public:
    P get_payload(uint8_t pos) {
        return payloads_[pos];
    }
    //shift by one for [begin, end]
    void shift_by_one(int begin, int end) {
        assert(end < this->key_n_ && end < (node_capacity - 1));
        for (int i = end; i >= begin; --i) {
            this->keys_[i+1] = this->keys_[i];
            payloads_[i+1] = payloads_[i];
        }
    }
    void shift_by_one(int begin) {
        assert(begin < this->key_n_);
        for (int i = this->key_n_ - 1; i >= begin; --i) {
            this->keys_[i+1] = this->keys_[i];
            payloads_[i+1] = payloads_[i];
        }
    }
};

template<class T, class P>
class DeltaIndex {
public:
    class DeltaNodeIterator;
    class DeltaKeyIterator;

    typedef DeltaNode<T, P> node_t;
    typedef DeltaInternal<T, P> internal_t;
    typedef DeltaLeaf<T, P> leaf_t;
    typedef DeltaIndex<T, P> delta_index_t;
    typedef DeltaNodeIterator delta_node_iterator_t;
    typedef DeltaKeyIterator delta_key_iterator_t;

public:
    //used for proper deletion of delta index.
    class DeltaNodeIterator {
    public:
        delta_index_t *index_ = nullptr;
        node_t* cur_node_ = nullptr;
        std::stack<node_t*> node_stack_;  // helps with traversal

        // Start with root as cur and all children of root in stack
        DeltaNodeIterator(delta_index_t* index)
            : index_(index), cur_node_(index->root_node_) {
            if (cur_node_ && !cur_node_->is_leaf_) {
                auto node = static_cast<internal_t*>(cur_node_);
                for (int i = 0; i < node->child_n_; ++i) {
                    node_stack_.push(node->childrens_[i]);
                }
            }
        }

        node_t* current() const { return cur_node_; }

        node_t* next() {
            if (node_stack_.empty()) {
                cur_node_ = nullptr;
                return nullptr;
            }

            cur_node_ = node_stack_.top();
            node_stack_.pop();

            if (!cur_node_->is_leaf_) {
                auto node = static_cast<internal_t*>(cur_node_);
                for (int i = 0; i < node->child_n_; ++i) {
                    node_stack_.push(node->childrens_[i]);
                }
            }

            return cur_node_;
        }

        bool is_end() const { return cur_node_ == nullptr; }
    };

    //for iteration through keys
public:
    class DeltaKeyIterator {
    public:
        delta_index_t *index_ = nullptr;
        leaf_t* cur_node_ = nullptr;
        uint8_t cur_idx_ = -1;

        DeltaKeyIterator(delta_index_t* index)
            : index_(index), cur_node_(index->leaf_begin_->next_leaf_) {
            if (index_->tot_key_n_ == 0) {cur_idx_ = -1;}
            else {cur_idx_ = 0;}
        }

        DeltaKeyIterator(delta_index_t* index, AlexKey<T>*key)
            : index_(index) {
            if (index_->tot_key_n_ == 0) {
                cur_idx_ = -1;
            } else if (key == nullptr) { //same as DeltaKeyIterator without key
                cur_node_ = index_->leaf_begin_->next_leaf_;
                cur_idx_ = 0;
            } else {
                cur_node_ = index->find_leaf(*key);
                cur_idx_ = cur_node_->find_first_larger_than_or_equal_to(*key);
            }
        }

        AlexKey<T> &key() {
            assert(cur_idx_ != -1);
            return cur_node_->get_key(cur_idx_);
        }

        P payload() {
            assert(cur_idx_ != -1);
            return cur_node_->get_payload(cur_idx_);
        }

        void next() {
            if (cur_idx_ == -1) {return;}
            cur_idx_++;
            if (cur_node_->key_n_ <= cur_idx_) {
                if (cur_node_->next_leaf_ == nullptr) {
                    cur_idx_ = -1;
                }
                else {
                    cur_node_ = cur_node_->next_leaf_;
                    if (cur_node_->key_n_ != 0) {cur_idx_ = 0;}
                    else {cur_idx_ = -1;}
                }
            }
        }

        void reinit() {
            cur_node_ = index_->leaf_begin_;
            if (cur_node_->key_n_ != 0) {cur_idx_ = 0;}
            else {cur_idx_ = -1;}
        }

        bool is_end() {
            return cur_idx_ == -1;
        }
    };

public:
    node_t *root_node_ = nullptr;
    leaf_t *leaf_begin_ = nullptr; //dummy
    int tot_key_n_ = 0;

    //lock happens when
    //1. just before insertion is about to happen on specific leaf node
    //2. before iterating through the whole tree on read.
    //different semantic compared to sindex. May hurt efficiency.
    pthread_rwlock_t delta_index_rw_lock_ = PTHREAD_RWLOCK_INITIALIZER;

public:
    DeltaIndex() {
        root_node_ = create_leaf(nullptr);
        leaf_begin_ = create_leaf(nullptr);
        leaf_begin_->next_leaf_ = static_cast<leaf_t *>(root_node_);
        leaf_begin_->prev_leaf_ = nullptr;
    }
    ~DeltaIndex() {
        for (DeltaNodeIterator node_it = DeltaNodeIterator(this); !node_it.is_end();
            node_it.next()) {
            delete_node(node_it.current());
        }
        delete_node(leaf_begin_);
        pthread_rwlock_destroy(&delta_index_rw_lock_);
    }
public:

    internal_t *create_internal(internal_t *parent) {
        return new DeltaInternal<T, P>(parent);
    }

    leaf_t *create_leaf(internal_t *parent) {
        return new DeltaLeaf<T, P>(parent);
    }

    void delete_node(node_t *node) {
        if (node == nullptr) {
            return;
        } else if (node->is_leaf_) {
            delete static_cast<leaf_t*>(node);
        } else {
            delete static_cast<internal_t*>(node);
        }
    }

    //inserts
    //0 : succeed
    int insert(const AlexKey<T> &key, P payload) {
        leaf_t *target_leaf = find_leaf(key);
        uint8_t pos = target_leaf->find_first_larger_than_or_equal_to(key);
        pthread_rwlock_wrlock(&delta_index_rw_lock_);
        if (pos < target_leaf->key_n_) { //need to shift.
            //note that we also allow duplicate
            target_leaf->shift_by_one(pos);
            target_leaf->keys_[pos] = key;
            target_leaf->payloads_[pos] = payload;
            target_leaf->key_n_++;
            tot_key_n_++;
        } else if (pos == target_leaf->key_n_) {
            //new insert at end
            target_leaf->keys_[pos] = key;
            target_leaf->payloads_[pos] = payload;
            target_leaf->key_n_++;
            tot_key_n_++;
        } else {abort();} //shouldn't happen.

        if (target_leaf->key_n_ == node_capacity) { //node on limit.
            split(target_leaf);
        }

        //note that we don't give any valid insert position.
        //I think we shouldn't allow iterating with iterator returned from insert in alex
        //maybe fix it next time...
        pthread_rwlock_unlock(&delta_index_rw_lock_);
        return 0;
    }

    //reading
    //bool : did read succeed?
    //P : If succeeded, payload related to that key.
    //if failed, 0 / 1 means locked / failure. (could fail if it just doesn't exist.)
    std::pair<bool, P> get_payload(const AlexKey<T> &key) {
        //needs to lock first. just try it. if failed, read later.
        if (pthread_rwlock_tryrdlock(&delta_index_rw_lock_)) {
            //failed obtaining lock
            return {false, 0};
        }

        leaf_t *target_leaf = find_leaf(key);
        uint8_t pos = target_leaf->find_first_larger_than_or_equal_to(key);
        if (pos < target_leaf->key_n_ && key == target_leaf->get_key(pos)) {
            pthread_rwlock_unlock(&delta_index_rw_lock_);
            return {true, target_leaf->get_payload(pos)};
        } else {//failed finding payload
            pthread_rwlock_unlock(&delta_index_rw_lock_);
            return {false, 1};
        }
    }
    
    //find leaf
    leaf_t *find_leaf(const AlexKey<T> &key) {
        node_t *cur_node = root_node_;
        
        while (true) {
            if (cur_node->is_leaf_) {
                return static_cast<leaf_t *>(cur_node);
            }
            cur_node = (static_cast<internal_t*>(cur_node))->find_child(key);
        }
    }

    //splitting procedure
    void split(leaf_t *target_leaf) {
        node_t *target_node = static_cast<node_t*>(target_leaf);
        internal_t *parent;
        while (true) { //recursive splitting

            if (target_node->parent_ == nullptr) {
                //this should be the last iteration.
                assert(target_node == root_node_);
                parent = create_internal(nullptr);  
                root_node_ = parent;
            } else {
                parent = target_node->parent_;
            }

            int idx = 0;
            int left_half_cnt = node_capacity/2;

            if (target_node->is_leaf_) {//full node was leaf
                leaf_t *target_leaf = static_cast<leaf_t*>(target_node);
                leaf_t *left_leaf = create_leaf(parent);
                leaf_t *right_leaf = create_leaf(parent);

                if (parent->key_n_ == 0) {
                    //new parent.
                    parent->key_n_++;
                    parent->childrens_[0] = left_leaf;
                    parent->childrens_[1] = right_leaf;
                    left_leaf->pos_in_childrens_ = 0;
                    right_leaf->pos_in_childrens_ = 1;
                    parent->keys_[0] = target_leaf->keys_[left_half_cnt];
                    parent->child_n_ = 2;
                } else {
                    //existing parent
                    //we first need to find the position of leaf that's begin splitted
                    //in that parent.
                    parent->key_n_++;
                    uint8_t target_pos_in_children = target_leaf->pos_in_childrens_;
                    
                    //1. shift childrens_ and keys_ by one
                    //shift start pos is right next to the pos where target existed
                    parent->shift_by_one(target_pos_in_children + 1);
                    
                    //2. there will be a new space in keys_. put middle key there.
                    parent->keys_[target_pos_in_children + 1] = target_leaf->keys_[left_half_cnt];

                    //3. update childrens_ with left_leaf and right_leaf
                    parent->childrens_[target_pos_in_children] = left_leaf;
                    parent->childrens_[target_pos_in_children + 1] = right_leaf;
                    left_leaf->pos_in_childrens_ = target_pos_in_children;
                    right_leaf->pos_in_childrens_ = target_pos_in_children + 1;
                    parent->child_n_ += 1;
                }

                //fill data
                left_leaf->key_n_ = left_half_cnt;
                right_leaf->key_n_ = left_half_cnt + 1;
                for (int i = 0; i < left_half_cnt; i++, idx++) {
                    left_leaf->keys_[i] = target_leaf->keys_[idx];
                    left_leaf->payloads_[i] = target_leaf->payloads_[idx];
                }
                for (int i = 0; i < left_half_cnt + 1; i++, idx++) {
                    right_leaf->keys_[i] = target_leaf->keys_[idx];
                    right_leaf->payloads_[i] = target_leaf->payloads_[idx];
                }
                
                //linking
                left_leaf->next_leaf_ = right_leaf;
                right_leaf->prev_leaf_ = left_leaf;
                right_leaf->next_leaf_ = target_leaf->next_leaf_;
                left_leaf->prev_leaf_ = target_leaf->prev_leaf_;
                if (target_leaf->next_leaf_ != nullptr) {
                    target_leaf->next_leaf_->prev_leaf_ = right_leaf;
                }
                if (target_leaf->prev_leaf_ != nullptr) {
                    target_leaf->prev_leaf_->next_leaf_ = left_leaf;
                }

                delete target_leaf;

            } else {//full node was internal
                internal_t *target_internal = static_cast<internal_t*>(target_node);
                internal_t *left_internal = create_internal(parent);
                internal_t *right_internal = create_internal(parent);

                if (parent->key_n_ == 0) {
                    //new parent.
                    parent->key_n_++;
                    parent->childrens_[0] = left_internal;
                    parent->childrens_[1] = right_internal;
                    left_internal->pos_in_childrens_ = 0;
                    right_internal->pos_in_childrens_ = 1;
                    parent->keys_[0] = target_internal->keys_[left_half_cnt];
                    parent->child_n_ = 2;
                } else {
                    //existing parent
                    //we first need to find the position of leaf that's begin splitted
                    //in that parent.
                    parent->key_n_++;
                    uint8_t target_pos_in_children = target_internal->pos_in_childrens_;
                    
                    //1. shift childrens_ and keys_ by one
                    //shift start pos is right next to the pos where target existed
                    parent->shift_by_one(target_pos_in_children + 1);
                    
                    //2. there will be a new space in keys_. put middle key there.
                    parent->keys_[target_pos_in_children + 1] = target_internal->keys_[left_half_cnt];

                    //3. update childrens_ with left_leaf and right_leaf
                    parent->childrens_[target_pos_in_children] = left_internal;
                    parent->childrens_[target_pos_in_children + 1] = right_internal;
                    left_internal->pos_in_childrens_ = target_pos_in_children;
                    right_internal->pos_in_childrens_ = target_pos_in_children + 1;
                    parent->child_n_ += 1;
                }

                //fill data
                left_internal->key_n_ = left_half_cnt;
                right_internal->key_n_ = left_half_cnt + 1;
                for (int i = 0; i < left_half_cnt; i++, idx++) {
                    left_internal->keys_[i] = target_internal->keys_[idx];
                    left_internal->childrens_[i] = target_internal->childrens_[idx];
                }
                for (int i = 0; i < left_half_cnt + 1; i++, idx++) {
                    right_internal->keys_[i] = target_internal->keys_[idx];
                    right_internal->childrens_[i] = target_internal->childrens_[idx];
                }

                delete target_internal;
                
            }

            if (parent->key_n_ < node_capacity) {break;}
            else {target_node = parent;}
        }
    }

};


}